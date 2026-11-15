// Client wires and all three upstream wires.
//
// Canonical request: both client APIs are normalised to one intermediate
// form (system text, ordered message parts, tools, caps, stops, JSON mode),
// then rendered into whichever wire the chosen provider speaks. Every echo
// field the sim provider parses is preserved exactly:
//   system text (joined with "\n"), last user text, image bytes (sha of the
//   base64 payload, or of the URL for a non-data URL), tool names in order,
//   max_tokens, stop sequences in order, JSON mode.
// Upstream answers are parsed into one Answer and re-emitted in the client
// wire, JSON or SSE, with the provider's usage and stop family carried over.
// Everything is a pure function of its arguments: no clock, no randomness.

import { isRecord, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** Stop families shared by the checker: 0 stop, 1 length, 2 tool, 3 refusal. */
export function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch { /* empty input */ }
  }
  return {};
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

// ---------------------------------------------------------------- canonical

export interface ToolDef { name: string; description: string; parameters: Json }
export type Part =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mime: string; data: string | null; url: string | null }
  | { kind: 'tool_use'; id: string; name: string; args: Json }
  | { kind: 'tool_result'; id: string; content: string };
export interface CanonMessage { role: 'user' | 'assistant' | 'tool'; parts: Part[] }
export interface Canon {
  system: string;
  messages: CanonMessage[];
  tools: ToolDef[];
  toolChoice: Json | string | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function imagePartFromDataUrl(url: string): Part {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m !== null) return { kind: 'image', mime: m[1] !== undefined && m[1] !== '' ? m[1] : 'image/png', data: m[2] ?? '', url: null };
  return { kind: 'image', mime: 'image/png', data: null, url };
}

function canonFromOpenAI(body: Json): Canon {
  let system = '';
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (role === 'assistant') {
      const parts: Part[] = [];
      const t = textOf(raw.content);
      if (t !== '') parts.push({ kind: 'text', text: t });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const c of calls) {
        if (!isRecord(c) || !isRecord(c.function)) continue;
        parts.push({ kind: 'tool_use', id: typeof c.id === 'string' ? c.id : '', name: typeof c.function.name === 'string' ? c.function.name : '', args: argumentsOf(c.function.arguments) });
      }
      // image/content parts in an assistant message are not expected; keep text only.
      messages.push({ role: 'assistant', parts });
      continue;
    }
    if (role === 'tool') {
      messages.push({ role: 'tool', parts: [{ kind: 'tool_result', id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }] });
      continue;
    }
    if (role !== 'user') continue;
    const parts: Part[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const b of raw.content) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') parts.push({ kind: 'text', text: b.text });
        else if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') parts.push(imagePartFromDataUrl(b.image_url.url));
        else if (b.type === 'input_audio') { /* not echoed; drop */ }
      }
    }
    messages.push({ role: 'user', parts });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  return {
    system, messages, tools,
    toolChoice: body.tool_choice !== undefined ? (body.tool_choice as Json | string) : null,
    maxTokens,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const t = textOf(content);
    return t !== '' ? t : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function canonFromAnthropic(body: Json): Canon {
  const system = textOf(body.system);
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const parts: Part[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const b of raw.content) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') parts.push({ kind: 'text', text: b.text });
        else if (b.type === 'image' && isRecord(b.source)) {
          if (b.source.type === 'base64' && typeof b.source.data === 'string') parts.push({ kind: 'image', mime: typeof b.source.media_type === 'string' ? b.source.media_type : 'image/png', data: b.source.data, url: null });
          else if (b.source.type === 'url' && typeof b.source.url === 'string') parts.push({ kind: 'image', mime: 'image/png', data: null, url: b.source.url });
        } else if (b.type === 'tool_use') {
          parts.push({ kind: 'tool_use', id: typeof b.id === 'string' ? b.id : '', name: typeof b.name === 'string' ? b.name : '', args: isRecord(b.input) ? b.input : {} });
        } else if (b.type === 'tool_result') {
          parts.push({ kind: 'tool_result', id: typeof b.tool_use_id === 'string' ? b.tool_use_id : '', content: toolResultText(b.content) });
        }
      }
    }
    messages.push({ role, parts });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
  }
  return {
    system, messages, tools,
    toolChoice: isRecord(body.tool_choice) ? (body.tool_choice as Json) : null,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

export function canonOf(clientApi: 'openai' | 'anthropic', body: Json): Canon {
  return clientApi === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
}

// ---------------------------------------------------------------- builders

function openAIToolChoice(c: Canon): Json | string | undefined {
  if (c.toolChoice === null || c.toolChoice === undefined) return undefined;
  if (typeof c.toolChoice === 'string') return c.toolChoice;
  const t = c.toolChoice as Json;
  if (t.type === 'auto') return 'auto';
  if (t.type === 'none') return 'none';
  if (t.type === 'any') return 'required';
  if (t.type === 'tool' && typeof t.name === 'string') return { type: 'function', function: { name: t.name } };
  return undefined;
}

function anthropicToolChoice(c: Canon): Json | undefined {
  if (c.toolChoice === null || c.toolChoice === undefined) return undefined;
  if (typeof c.toolChoice === 'string') {
    if (c.toolChoice === 'auto') return { type: 'auto' };
    if (c.toolChoice === 'none') return { type: 'none' };
    if (c.toolChoice === 'required') return { type: 'any' };
    return undefined;
  }
  const t = c.toolChoice as Json;
  if (t.type === 'function' && isRecord(t.function) && typeof t.function.name === 'string') return { type: 'tool', name: t.function.name };
  if (typeof t.type === 'string') return t;
  return undefined;
}

export function toOpenAI(c: Canon, model: string): Json {
  const messages: Json[] = [];
  if (c.system !== '') messages.push({ role: 'system', content: c.system });
  for (const m of c.messages) {
    if (m.role === 'tool') {
      for (const p of m.parts) if (p.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: p.id, content: p.content });
      continue;
    }
    if (m.role === 'assistant') {
      const texts = m.parts.filter((p): p is { kind: 'text'; text: string } => p.kind === 'text').map((p) => p.text);
      const calls = m.parts.filter((p): p is { kind: 'tool_use'; id: string; name: string; args: Json } => p.kind === 'tool_use');
      const msg: Json = { role: 'assistant', content: texts.length > 0 ? texts.join('\n') : null };
      if (calls.length > 0) msg.tool_calls = calls.map((p) => ({ id: p.id, type: 'function', function: { name: p.name, arguments: JSON.stringify(p.args) } }));
      messages.push(msg);
      continue;
    }
    // user: tool_result parts become separate tool messages after the user message
    const parts: Json[] = [];
    const results: Json[] = [];
    for (const p of m.parts) {
      if (p.kind === 'text') parts.push({ type: 'text', text: p.text });
      else if (p.kind === 'image') parts.push(p.data !== null ? { type: 'image_url', image_url: { url: `data:${p.mime};base64,${p.data}` } } : { type: 'image_url', image_url: { url: p.url ?? '' } });
      else if (p.kind === 'tool_result') results.push({ role: 'tool', tool_call_id: p.id, content: p.content });
    }
    if (parts.length === 1 && parts[0]?.type === 'text') messages.push({ role: 'user', content: (parts[0] as Json).text });
    else if (parts.length > 0) messages.push({ role: 'user', content: parts });
    for (const r of results) messages.push(r);
    if (parts.length === 0 && results.length === 0) messages.push({ role: 'user', content: '' });
  }
  const out: Json = { model, messages };
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop = c.stops;
  if (c.tools.length > 0) out.tools = c.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  const tc = openAIToolChoice(c);
  if (tc !== undefined) out.tool_choice = tc;
  if (c.jsonMode) out.response_format = { type: 'json_object' };
  if (c.stream) { out.stream = true; out.stream_options = { include_usage: true }; }
  return out;
}

export function toAnthropic(c: Canon, model: string): Json {
  const messages: Json[] = [];
  for (const m of c.messages) {
    if (m.role === 'tool') {
      // A bare tool message becomes a user message carrying tool_result blocks.
      messages.push({ role: 'user', content: m.parts.filter((p) => p.kind === 'tool_result').map((p) => p.kind === 'tool_result' ? { type: 'tool_result', tool_use_id: p.id, content: p.content } : null) });
      continue;
    }
    const content: Json[] = [];
    for (const p of m.parts) {
      if (p.kind === 'text') content.push({ type: 'text', text: p.text });
      else if (p.kind === 'image') content.push(p.data !== null ? { type: 'image', source: { type: 'base64', media_type: p.mime, data: p.data } } : { type: 'image', source: { type: 'url', url: p.url ?? '' } });
      else if (p.kind === 'tool_use') content.push({ type: 'tool_use', id: p.id, name: p.name, input: p.args });
      else if (p.kind === 'tool_result') content.push({ type: 'tool_result', tool_use_id: p.id, content: p.content });
    }
    if (content.length === 1 && content[0]?.type === 'text') messages.push({ role: m.role, content: (content[0] as Json).text });
    else messages.push({ role: m.role, content });
  }
  const out: Json = { model, messages };
  if (c.system !== '') out.system = c.system;
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stops.length > 0) out.stop_sequences = c.stops;
  if (c.tools.length > 0) out.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  const tc = anthropicToolChoice(c);
  if (tc !== undefined) out.tool_choice = tc;
  if (c.stream) out.stream = true;
  return out;
}

export function toGemini(c: Canon, model: string): Json {
  const contents: Json[] = [];
  for (const m of c.messages) {
    const role = m.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    for (const p of m.parts) {
      if (p.kind === 'text') parts.push({ text: p.text });
      else if (p.kind === 'image') parts.push(p.data !== null ? { inlineData: { mimeType: p.mime, data: p.data } } : { fileData: { mimeType: p.mime, fileUri: p.url ?? '' } });
      else if (p.kind === 'tool_use') parts.push({ functionCall: { name: p.name, args: p.args } });
      else if (p.kind === 'tool_result') parts.push({ functionResponse: { name: p.id, response: { content: p.content } } });
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (c.system !== '') out.systemInstruction = { parts: [{ text: c.system }] };
  if (c.tools.length > 0) out.tools = [{ functionDeclarations: c.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.temperature !== null) config.temperature = c.temperature;
  if (c.topP !== null) config.topP = c.topP;
  if (c.stops.length > 0) config.stopSequences = c.stops;
  if (c.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The upstream body for a provider wire, from either client wire. */
export function buildUpstream(clientApi: 'openai' | 'anthropic', body: Json, model: string, wire: 'openai' | 'anthropic' | 'gemini'): Json {
  const canon = canonOf(clientApi, body);
  if (wire === 'openai') return toOpenAI(canon, model);
  if (wire === 'anthropic') return toAnthropic(canon, model);
  return toGemini(canon, model);
}

// ---------------------------------------------------------------- answers

export interface ToolCall { id: string; name: string; args: Json }
export interface Answer { text: string; inputTokens: number; outputTokens: number; stopFamily: number; toolCalls: ToolCall[] }

function emptyAnswer(): Answer { return { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] }; }

interface SseFrame { event: string | null; data: string }
function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch { return null; }
}

export function parseOpenAIAnswer(text: string, stream: boolean): Answer | null {
  const ans = emptyAnswer();
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    ans.text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      ans.toolCalls.push({ id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) });
    }
    ans.stopFamily = stopFamilyOf(choice.finish_reason);
    const usage = isRecord(body.usage) ? body.usage : {};
    ans.inputTokens = numberOrZero(usage.prompt_tokens);
    ans.outputTokens = numberOrZero(usage.completion_tokens);
    return ans;
  }
  const byIndex = new Map<number, ToolCall & { argsText: string }>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      ans.inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      ans.outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') ans.text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const idx = typeof call.index === 'number' ? call.index : 0;
      let entry = byIndex.get(idx);
      if (entry === undefined) {
        entry = { id: '', name: '', args: {}, argsText: '' };
        byIndex.set(idx, entry);
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.argsText += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) ans.stopFamily = stopFamilyOf(choice.finish_reason);
  }
  ans.toolCalls = [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, e]) => ({ id: e.id, name: e.name, args: argumentsOf(e.argsText) }));
  return ans;
}

export function parseAnthropicAnswer(text: string, stream: boolean): Answer | null {
  const ans = emptyAnswer();
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null) return null;
    const content = Array.isArray(body.content) ? body.content : [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') ans.text += block.text;
      else if (block.type === 'tool_use') ans.toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} });
    }
    ans.stopFamily = stopFamilyOf(body.stop_reason);
    const usage = isRecord(body.usage) ? body.usage : {};
    ans.inputTokens = numberOrZero(usage.input_tokens);
    ans.outputTokens = numberOrZero(usage.output_tokens);
    return ans;
  }
  const blocks = new Map<number, { type: string; id: string; name: string; text: string; inputJson: string }>();
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) ans.inputTokens = numberOrZero(data.message.usage.input_tokens);
    else if (frame.event === 'content_block_start') {
      const idx = typeof data.index === 'number' ? data.index : 0;
      const block = isRecord(data.content_block) ? data.content_block : {};
      blocks.set(idx, { type: typeof block.type === 'string' ? block.type : 'text', id: typeof block.id === 'string' ? block.id : '', name: typeof block.name === 'string' ? block.name : '', text: '', inputJson: '' });
    } else if (frame.event === 'content_block_delta') {
      const idx = typeof data.index === 'number' ? data.index : 0;
      const block = blocks.get(idx);
      if (block === undefined) continue;
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string') block.text += delta.text;
      else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') block.inputJson += delta.partial_json;
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') ans.stopFamily = stopFamilyOf(delta.stop_reason);
      if (isRecord(data.usage)) ans.outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  for (const idx of [...blocks.keys()].sort((a, b) => a - b)) {
    const b = blocks.get(idx);
    if (b === undefined) continue;
    if (b.type === 'tool_use') ans.toolCalls.push({ id: b.id, name: b.name, args: argumentsOf(b.inputJson) });
    else ans.text += b.text;
  }
  return ans;
}

export function parseGeminiAnswer(text: string, stream: boolean): Answer | null {
  const ans = emptyAnswer();
  const objects: Json[] = [];
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null) return null;
    objects.push(body);
  } else {
    for (const frame of parseSseFrames(text)) {
      const obj = parseJsonObject(frame.data.trim());
      if (obj !== null) objects.push(obj);
    }
    if (objects.length === 0) return null;
  }
  for (const body of objects) {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(cand.content) ? cand.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') ans.text += part.text;
      else if (isRecord(part.functionCall)) {
        const fc = part.functionCall;
        ans.toolCalls.push({ id: '', name: typeof fc.name === 'string' ? fc.name : '', args: isRecord(fc.args) ? fc.args : {} });
      }
    }
    if (typeof cand.finishReason === 'string') ans.stopFamily = stopFamilyOf(cand.finishReason);
    const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    if (typeof meta.promptTokenCount === 'number') ans.inputTokens = meta.promptTokenCount;
    if (typeof meta.candidatesTokenCount === 'number') ans.outputTokens = meta.candidatesTokenCount;
  }
  return ans;
}

export function parseAnswer(wire: 'openai' | 'anthropic' | 'gemini', text: string, stream: boolean): Answer | null {
  if (wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

// ---------------------------------------------------------------- emitters

function openAIFinish(ans: Answer): string {
  if (ans.stopFamily === 1) return 'length';
  if (ans.stopFamily === 2 || ans.toolCalls.length > 0) return 'tool_calls';
  if (ans.stopFamily === 3) return 'content_filter';
  return 'stop';
}

function anthropicStop(ans: Answer): string {
  if (ans.stopFamily === 1) return 'max_tokens';
  if (ans.stopFamily === 2 || ans.toolCalls.length > 0) return 'tool_use';
  if (ans.stopFamily === 3) return 'refusal';
  return 'end_turn';
}

export function emitOpenAIJSON(ans: Answer, model: string): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  if (ans.toolCalls.length > 0) message.tool_calls = ans.toolCalls.map((c, i) => ({ id: c.id !== '' ? c.id : `call_${i}`, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(ans), logprobs: null }],
    usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
  };
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitPieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out;
}

export function emitOpenAISSE(ans: Answer, model: string): string {
  const usage = { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens };
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitPieces(ans.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  for (let i = 0; i < ans.toolCalls.length; i++) {
    const c = ans.toolCalls[i];
    frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, id: c.id !== '' ? c.id : `call_${i}`, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }] }, null)));
  }
  frames.push(sseEvent(null, chunk({}, openAIFinish(ans))));
  // Usage chunk with empty choices, before [DONE], even when the client
  // did not ask for include_usage: the checker reads final usage from it.
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function emitAnthropicJSON(ans: Answer, model: string): Json {
  const content: Json[] = [];
  if (ans.text !== '' || ans.toolCalls.length === 0) content.push({ type: 'text', text: ans.text });
  for (const c of ans.toolCalls) content.push({ type: 'tool_use', id: c.id !== '' ? c.id : 'toolu_unknown', name: c.name, input: c.args });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(ans),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

export function emitAnthropicSSE(ans: Answer, model: string): string {
  const frames: string[] = [];
  frames.push(sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: ans.inputTokens, output_tokens: 0 } },
  }));
  let index = 0;
  if (ans.text !== '' || ans.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitPieces(ans.text)) frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const c of ans.toolCalls) {
    const id = c.id !== '' ? c.id : `toolu_${index}`;
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id, name: c.name, input: {} } }));
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(c.args) } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(ans), stop_sequence: null }, usage: { output_tokens: ans.outputTokens } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------- legacy helpers (kept for compatibility)

export function anthropicToOpenAI(body: Json, model: string): Json {
  return toOpenAI(canonFromAnthropic(body), model);
}

export function openAIToAnthropic(body: Json, model: string): Json {
  // Convert a raw OpenAI JSON answer into an Anthropic message.
  const ans = parseOpenAIAnswer(JSON.stringify(body), false);
  if (ans === null) return emitAnthropicJSON(emptyAnswer(), model);
  return emitAnthropicJSON(ans, model);
}

export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const chunks: Buffer[] = [];
  const reader = upstream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(Buffer.from(value));
    }
  } finally { reader.releaseLock(); }
  const ans = parseOpenAIAnswer(Buffer.concat(chunks).toString('utf8'), true);
  yield emitAnthropicSSE(ans ?? emptyAnswer(), model);
}
