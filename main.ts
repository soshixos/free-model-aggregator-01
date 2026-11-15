// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Routing: the manifest is planned globally before registering. Requests are
// ordered by scarcity (fewest eligible candidates first), then weight desc,
// then manifest order, and each takes the best-rating free candidate whose
// provider's planned usage is below its conservative budget cap. At runtime
// the planned candidate is tried first, then the remaining candidates by
// rating (providers still under their cap first), failing over on any
// 402/429/5xx or fetch error. The upstream response is buffered fully
// before the client is answered, so a failed upstream never reaches it.
// Per-provider call counts and consecutive failures are tracked: after 2
// consecutive failures a provider is dead for the rest of the trial (unless
// it is the only candidate left), and after calls_max it is exhausted.
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

/** Conservative per-trial budget caps used for planning (below true budgets). */
const CAP: Record<string, number> = { groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10, sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12 };
/** Call counts at which a provider is treated as exhausted. */
const CALLS_MAX: Record<string, number> = { groq: 12, cerebras: 6, gemini: 8, zai: 5, openrouter: 12, sambanova: 5, pollinations: 14, cloudflare: 8, ovh: 6, kilol: 16 };

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
    const under = sorted.filter((c) => planUsageOf(c.probe.provider.id) < (CAP[c.probe.provider.id] ?? 0));
    const pick = (under.length > 0 ? under : sorted)[0];
    if (pick === undefined) continue;
    planned.set(index, pick);
    plannedUsage.set(pick.probe.provider.id, planUsageOf(pick.probe.provider.id) + 1);
  }
}

// ---- runtime state -------------------------------------------------------

const calls = new Map<string, number>();
const consecutiveFailures = new Map<string, number>();
const dead = new Set<string>();
const exhausted = new Set<string>();
const callsOf = (pid: string): number => calls.get(pid) ?? 0;

function markFailure(pid: string): void {
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

/** Runtime order: planned candidate first, then the rest by rating, under-cap providers first. */
function runtimeOrder(requires: Requirement, jsonMode: boolean, plannedCand: Candidate | null): Candidate[] {
  const cands = candidatesFor(requires, jsonMode);
  const rest = cands.filter((c) => plannedCand === null || candidateKey(c) !== candidateKey(plannedCand));
  rest.sort((a, b) => {
    const aUnder = callsOf(a.probe.provider.id) < (CAP[a.probe.provider.id] ?? 0) ? 0 : 1;
    const bUnder = callsOf(b.probe.provider.id) < (CAP[b.probe.provider.id] ?? 0) ? 0 : 1;
    if (aUnder !== bUnder) return aUnder - bUnder;
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
