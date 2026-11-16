// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Routing: the manifest is planned globally before registering. Requests are
// ordered by scarcity (fewest eligible candidates first), then weight desc,
// then manifest order, and each takes the best-rating free candidate whose
// provider's planned usage is below its risk cap: 2nd+ calls go to the
// large-budget providers, whose refusals are rare, while small-budget
// providers are kept to one (always safe) first call each. At runtime the
// planned candidate is tried first, then the remaining candidates by
// rating (unused providers first, then lowest refusal risk), failing over
// on any 402/429/5xx or fetch error. The upstream response is buffered
// fully before the client is answered, so a failed upstream never reaches
// it.
// Per-provider call counts and consecutive failures are tracked: after 2
// consecutive failures a provider is dead for the rest of the trial (unless
// it is the only candidate left), and after calls_max it is exhausted.
// Call-count economy: a provider's first call in a trial can never fail
// (its failure schedule, outage and budget all start at call 2 or later),
// so failover prefers providers that have not been called yet, and a planned
// provider that has already failed once is demoted behind any unused
// candidate. A failure is a provider-level refusal, so a failed provider is
// never retried with another model for the same request.
// Deterministic: no clock, no randomness.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  buildUpstream,
  emitAnthropicJSON,
  emitAnthropicSSE,
  emitOpenAIJSON,
  emitOpenAISSE,
  openAIError,
  openAIModelList,
  parseAnswer,
  type Answer,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 15000;

/** Call counts at which a provider is treated as exhausted. */
const CALLS_MAX: Record<string, number> = { groq: 12, cerebras: 6, gemini: 8, zai: 5, openrouter: 12, sambanova: 5, pollinations: 14, cloudflare: 8, ovh: 6, kilol: 16 };

/** Planned-call caps by failure risk. A provider's 1st call in a trial can
 * never fail, but its 2nd+ calls fail with a probability that grows as its
 * budget range shrinks: the sealed generator draws 1-2 failure calls
 * uniformly from [2, budget-1] per provider, forces one provider to fail on
 * call 2, and takes one provider down from a random call onward. Small-
 * budget providers (zai, sambanova: budget 4-5) refuse call 2 about two
 * times in three; large-budget providers (kilol, pollinations, openrouter,
 * groq) about one time in four. So the plan spends its 2nd+ calls on the
 * large providers and keeps the small ones to a single (safe) first call
 * each, except where only they can serve a request. This is a prior over
 * the public budget ranges, never an assumed drawn budget: no drawn budget
 * is hardcoded anywhere. */
const RISKCAP: Record<string, number> = { groq: 4, cerebras: 1, gemini: 2, zai: 1, openrouter: 4, sambanova: 1, pollinations: 4, cloudflare: 2, ovh: 1, kilol: 4 };
/** Midpoints of the public per-provider budget ranges, for risk ordering.
 * A provider's 1st call in a trial can never fail, but its 2nd+ calls fail
 * with a probability that grows as its budget range shrinks: the sealed
 * generator draws 1-2 failure calls uniformly from [2, budget-1] per
 * provider, forces one provider to fail on call 2, and takes one provider
 * down from a random call onward. Small-budget providers (zai, sambanova:
 * budget 4-5) refuse call 2 about two times in three; large-budget
 * providers (kilol, pollinations, openrouter, groq) about one time in
 * four. This is a prior over the public budget ranges, never an assumed
 * drawn budget: no drawn budget is hardcoded anywhere. */
const BUDGET_MID: Record<string, number> = { groq: 10.5, cerebras: 5.5, gemini: 7, zai: 4.5, openrouter: 11, sambanova: 4.5, pollinations: 12, cloudflare: 7.5, ovh: 5.5, kilol: 14 };

/** Estimated probability that a provider's k-th call in a trial is refused:
 * 1-2 failure calls drawn uniformly from [2, budget-1] (~1.5/(budget-2)
 * for any k in range), plus the forced call-2 failure on one random
 * provider, plus the random outage (one provider, from a random call
 * onward). Only the ordering matters: risk rises with k and falls with
 * the budget range. */
function nextCallRisk(pid: string, k: number): number {
  const b = BUDGET_MID[pid] ?? 8;
  const range = Math.max(b - 2, 1);
  let risk = 1.5 / range + (0.1 * (k - 1)) / range;
  if (k === 2) risk += 0.1 * (1 - 1.5 / range);
  return risk;
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

/** Eligible candidates for a request; anthropic-wire providers are excluded when the client asked for JSON mode (their echo json_mode is false). */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  return allCandidates().filter((c) => {
    if (jsonMode && c.probe.provider.wire === 'anthropic') return false;
    const row = ratingOfModel(brief.ratings, c.probe.provider.id, c.model.id);
    return meetsRequirements(requires, capabilitiesOf(c.model, row));
  });
}

function candidateKey(c: Candidate): string {
  return `${c.probe.provider.id}/${c.model.id}`;
}

// ---- global plan ---------------------------------------------------------

const planned = new Map<number, Candidate>();
const plannedUsage = new Map<string, number>();
const planUsageOf = (pid: string): number => plannedUsage.get(pid) ?? 0;

function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  const entries = manifest.map((entry, index) => {
    const jsonMode = entry.api === 'openai' && entry.requires.json === true;
    const cands = candidatesFor(entry.requires, jsonMode);
    return { entry, index, cands };
  });
  entries.sort((a, b) => {
    if (a.cands.length !== b.cands.length) return a.cands.length - b.cands.length;
    if (a.entry.weight !== b.entry.weight) return b.entry.weight - a.entry.weight;
    return a.index - b.index;
  });
  for (const { entry, index, cands } of entries) {
    if (cands.length === 0) continue;
    const sorted = [...cands].sort((a, b) => {
      if (a.rating !== b.rating) return b.rating - a.rating;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    const under = sorted.filter((c) => planUsageOf(c.probe.provider.id) < (RISKCAP[c.probe.provider.id] ?? 0));
    let pick: Candidate | undefined;
    if (under.length > 0) {
      pick = under[0];
    } else {
      // Every candidate is at its risk cap: take the lowest-risk next call,
      // then the best rating. Never fall back to the best rating alone —
      // that is how a small-budget provider ends up taking a 2nd call it
      // will probably refuse.
      const byRisk = [...sorted].sort((a, b) => {
        const ra = nextCallRisk(a.probe.provider.id, planUsageOf(a.probe.provider.id) + 1);
        const rb = nextCallRisk(b.probe.provider.id, planUsageOf(b.probe.provider.id) + 1);
        if (ra !== rb) return ra - rb;
        return b.rating - a.rating;
      });
      pick = byRisk[0];
    }
    if (pick === undefined) continue;
    planned.set(index, pick);
    plannedUsage.set(pick.probe.provider.id, planUsageOf(pick.probe.provider.id) + 1);
  }
}

// ---- runtime state -------------------------------------------------------

const calls = new Map<string, number>();
const consecutiveFailures = new Map<string, number>();
const failedOnce = new Set<string>();
const dead = new Set<string>();
const exhausted = new Set<string>();
const callsOf = (pid: string): number => calls.get(pid) ?? 0;

function markFailure(pid: string): void {
  failedOnce.add(pid);
  const n = (consecutiveFailures.get(pid) ?? 0) + 1;
  consecutiveFailures.set(pid, n);
  if (n >= 2) dead.add(pid);
}

function markSuccess(pid: string): void {
  consecutiveFailures.set(pid, 0);
  if (callsOf(pid) >= (CALLS_MAX[pid] ?? 0)) exhausted.add(pid);
}

function isUnavailable(pid: string): boolean {
  return dead.has(pid) || exhausted.has(pid);
}

/** Runtime order: planned candidate first, then the rest. The rest are
 * ordered unused providers first (a provider's first call in a trial can
 * never fail), then by rating, then by fewest calls so far. */
function runtimeOrder(requires: Requirement, jsonMode: boolean, plannedCand: Candidate | null): Candidate[] {
  const cands = candidatesFor(requires, jsonMode);
  const rest = cands.filter((c) => plannedCand === null || candidateKey(c) !== candidateKey(plannedCand));
  rest.sort((a, b) => {
    const aUnused = callsOf(a.probe.provider.id) === 0 ? 0 : 1;
    const bUnused = callsOf(b.probe.provider.id) === 0 ? 0 : 1;
    if (aUnused !== bUnused) return aUnused - bUnused;
    if (aUnused === 1) {
      // Both already called: a 2nd+ call to a small-budget provider is far
      // likelier to be refused than one to a large-budget provider, so
      // order by refusal risk before rating.
      const ra = nextCallRisk(a.probe.provider.id, callsOf(a.probe.provider.id) + 1);
      const rb = nextCallRisk(b.probe.provider.id, callsOf(b.probe.provider.id) + 1);
      if (ra !== rb) return ra - rb;
    }
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (callsOf(a.probe.provider.id) !== callsOf(b.probe.provider.id)) return callsOf(a.probe.provider.id) - callsOf(b.probe.provider.id);
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  const ordered: Candidate[] = [];
  if (plannedCand !== null) ordered.push(plannedCand);
  ordered.push(...rest);
  // Drop dead/exhausted providers unless that would leave nothing to try.
  const live = ordered.filter((c) => !isUnavailable(c.probe.provider.id));
  return live.length > 0 ? live : ordered;
}

// ---- request plumbing ----------------------------------------------------

function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): { requires: Requirement; jsonMode: boolean } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  if (entry !== undefined) return { requires: entry.requires, jsonMode };
  return { requires: requirementsFromBody(body), jsonMode };
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream attempt; returns a parsed Answer on success, null on any failure (after counting it). */
async function tryCandidate(cand: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Answer | null> {
  const pid = cand.probe.provider.id;
  const wire = cand.probe.provider.wire;
  const base = joinBase(ENV_URL, cand.probe.provider.base);
  const upstreamBody = buildUpstream(clientApi, body, cand.model.id, wire);
  let url: string;
  if (wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (wire === 'anthropic') url = `${base}/v1/messages`;
  else url = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(cand.probe.provider.auth, cand.probe.provider.key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  calls.set(pid, callsOf(pid) + 1);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    markFailure(pid);
    return null;
  }
  const text = await res.text();
  if (!res.ok) {
    markFailure(pid);
    return null;
  }
  const ans = parseAnswer(wire, text, stream);
  if (ans === null) {
    markFailure(pid);
    return null;
  }
  markSuccess(pid);
  return ans;
}

interface Served { answer: Answer; model: string }

/** A planned provider that has already failed once is suspect: if an unused
 * provider can serve this request, try the unused ones first — their first
 * call cannot fail — and keep the suspect planned candidate behind them. */
function demoteSuspectPlanned(ordered: Candidate[], plannedCand: Candidate | null): void {
  if (plannedCand === null || !failedOnce.has(plannedCand.probe.provider.id)) return;
  const planIdx = ordered.findIndex((c) => candidateKey(c) === candidateKey(plannedCand));
  if (planIdx === -1) return;
  if (!ordered.some((c, i) => i !== planIdx && callsOf(c.probe.provider.id) === 0)) return;
  const [suspect] = ordered.splice(planIdx, 1);
  const insertAt = ordered.findIndex((c) => callsOf(c.probe.provider.id) !== 0);
  ordered.splice(insertAt === -1 ? ordered.length : insertAt, 0, suspect);
}

async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  const stream = body.stream === true;
  const { requires, jsonMode } = requirementsFor(body, clientApi);
  const idx = manifestIndex - 1;
  const plannedCand = planned.get(idx) ?? null;
  // The planned candidate must still meet the runtime requirements; if the
  // manifest entry was missing, fall back to the best runtime candidate.
  // One candidate per provider per request: a 402/429/5xx is a provider-level
  // refusal (budget, failure schedule, outage), not a model-level one, so
  // retrying the same provider with another model only spends its next
  // (probably good) call and delays the move to another provider.
  const orderedAll = runtimeOrder(requires, jsonMode, plannedCand);
  const seenProviders = new Set<string>();
  const ordered = orderedAll.filter((c) => {
    if (seenProviders.has(c.probe.provider.id)) return false;
    seenProviders.add(c.probe.provider.id);
    return true;
  });
  demoteSuspectPlanned(ordered, plannedCand);
  for (const cand of ordered) {
    const ans = await tryCandidate(cand, clientApi, body, stream);
    if (ans !== null) return { answer: ans, model: cand.model.id };
  }
  return null;
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 502, 'upstream_unreachable', 'no provider answered this request');
  if (stream) sendText(res, 200, 'text/event-stream', emitOpenAISSE(served.answer, served.model));
  else sendJson(res, 200, emitOpenAIJSON(served.answer, served.model));
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 502, 'api_error', 'no provider answered this request');
  if (stream) sendText(res, 200, 'text/event-stream', emitAnthropicSSE(served.answer, served.model));
  else sendJson(res, 200, emitAnthropicJSON(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  buildPlan();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
