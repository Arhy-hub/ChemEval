// Juno's only server route. It passes the current turn to Claude and returns the reply.
// It does NOT store or log any message content. No passcode or key ever reaches it.
import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import { z } from "zod";
import { SECTIONS, turnSchema, type SectionId, type TurnResult } from "@/lib/schema";
import { JUNO_SYSTEM, sectionBrief } from "@/lib/juno-prompt";
import { mockTurn } from "@/lib/mock-interviewer";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MODEL = "claude-sonnet-5-5";
const MAX_HISTORY = 12; // only the recent turns of the current section are sent

const Body = z.object({
  name: z.string().max(60),
  section: z.enum(SECTIONS.map((s) => s.id) as [SectionId, ...SectionId[]]),
  draft: z.unknown(),
  messages: z.array(z.object({ from: z.enum(["juno", "user"]), text: z.string().max(2000) })).max(60),
  context: z.record(z.string(), z.string().nullable()).optional(),
  resumed: z.boolean().optional(),
});

const client = process.env.ANTHROPIC_API_KEY ? new Anthropic() : null;

export async function POST(req: Request) {
  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "bad_request" }, { status: 400 });
  const body = parsed.data;
  const history = body.messages.slice(-MAX_HISTORY);

  if (!client) {
    const lastUser = history.at(-1)?.from === "user" ? history.at(-1)!.text : null;
    const result = mockTurn(body.section, body.draft as never, lastUser);
    return Response.json({ ...result, mock: true } satisfies TurnResult & { mock: boolean });
  }

  // Claude needs the conversation to start with a user turn.
  const messages: Anthropic.Beta.BetaMessageParam[] = history.map((m) => ({
    role: m.from === "user" ? "user" : "assistant",
    content: m.text,
  }));
  if (messages[0]?.role !== "user") messages.unshift({ role: "user", content: "(Patient is ready.)" });
  if (messages.at(-1)?.role !== "user") messages.push({ role: "user", content: "(Continue.)" });

  try {
    const response = await client.beta.messages.parse({
      model: MODEL,
      max_tokens: 2000,
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system: [
        { type: "text", text: JUNO_SYSTEM, cache_control: { type: "ephemeral" } },
        { type: "text", text: sectionBrief({ ...body, draft: body.draft ?? {} }) },
      ],
      messages,
      output_config: { effort: "low", format: betaZodOutputFormat(turnSchema(body.section)) },
    });

    if (response.stop_reason === "refusal" || !response.parsed_output) {
      return Response.json({ error: "no_reply" }, { status: 502 });
    }
    return Response.json(response.parsed_output as TurnResult);
  } catch (err) {
    // Log only the error type/status, never the request content.
    const status = err instanceof Anthropic.APIError ? err.status : undefined;
    console.error("Claude call failed", status ?? (err as Error).name);
    const code = err instanceof Anthropic.RateLimitError ? 429 : 502;
    return Response.json({ error: "ai_unavailable" }, { status: code });
  }
}
