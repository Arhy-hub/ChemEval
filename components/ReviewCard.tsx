"use client";

// "Here's what I've got — is this right?" Shown after each section. Nothing is saved until "Looks good".
import { useState } from "react";
import { FIELD_LABELS, SECTIONS, isListSection, type AnySectionData, type SectionId } from "@/lib/schema";

type Item = Record<string, string | null>;

export function ReviewCard({
  section,
  data,
  saving,
  onConfirm,
  onFix,
}: {
  section: SectionId;
  data: AnySectionData;
  saving?: boolean;
  onConfirm: (data: AnySectionData) => void;
  onFix: () => void;
}) {
  const [draft, setDraft] = useState<AnySectionData>(data);
  const title = SECTIONS.find((s) => s.id === section)!.title;

  return (
    <div className="animate-sheet mt-auto max-h-full overflow-y-auto rounded-t-[28px] border-t border-line bg-night-2/95 px-5 pt-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] backdrop-blur-xl">
      <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-white/15" />
      <p className="text-xs font-semibold tracking-wider text-juno-300 uppercase">{title}</p>
      <h2 className="mt-1 text-lg font-semibold">Here&apos;s what I&apos;ve got. Is this right?</h2>
      <p className="mt-0.5 text-xs text-white/45">Tap anything to change it.</p>

      <div className="mt-4 divide-y divide-white/5 overflow-hidden rounded-2xl bg-white/[0.04] ring-1 ring-white/5">
        {isListSection(section) ? (
          <ListRows section={section} items={(draft as { items: Item[] }).items} onChange={(items) => setDraft({ items } as AnySectionData)} />
        ) : (
          Object.entries(draft as Item).map(([key, value]) =>
            key === "pregnancy" && value == null ? null : (
              <EditableRow
                key={key}
                label={FIELD_LABELS[key] ?? key}
                value={value}
                onChange={(v) => setDraft({ ...(draft as Item), [key]: v } as AnySectionData)}
              />
            ),
          )
        )}
      </div>

      <div className="mt-5 flex gap-3">
        <button onClick={onFix} disabled={saving} className="flex-1 rounded-full bg-white/[0.07] py-3.5 text-sm font-semibold text-white/80 ring-1 ring-white/10 transition active:scale-[0.97]">
          Something&apos;s wrong
        </button>
        <button
          onClick={() => onConfirm(draft)}
          disabled={saving}
          className="flex-[1.3] rounded-full bg-juno-400 py-3.5 text-sm font-bold text-night shadow-[0_0_24px_rgba(79,199,176,0.35)] transition active:scale-[0.97] disabled:opacity-60"
        >
          {saving ? "Saving…" : "Looks good ✓"}
        </button>
      </div>
    </div>
  );
}

const LIST_FIELDS: Record<string, { label: string; main: string; extra: string[] }> = {
  conditions: { label: "Condition", main: "name", extra: ["status", "notes"] },
  medications: { label: "Medicine", main: "name", extra: ["dose", "frequency", "notes"] },
  allergies: { label: "Allergy", main: "substance", extra: ["reaction"] },
  procedures: { label: "Operation or stay", main: "name", extra: ["when", "notes"] },
  family: { label: "Relative", main: "condition", extra: [] },
};

function ListRows({ section, items, onChange }: { section: SectionId; items: Item[]; onChange: (items: Item[]) => void }) {
  const f = LIST_FIELDS[section];
  if (!items.length) return <p className="px-4 py-3.5 text-sm text-white/60">None reported</p>;
  return (
    <>
      {items.map((item, i) => (
        <div key={i} className="flex items-start gap-2 pr-2">
          <div className="min-w-0 flex-1">
            <EditableRow
              label={section === "family" ? (item.relative ?? "Relative") : f.label}
              value={item[f.main]}
              sub={f.extra.map((k) => item[k]).filter((x) => x && x !== "unknown").join(" · ")}
              onChange={(v) => onChange(items.map((it, j) => (j === i ? { ...it, [f.main]: v } : it)))}
            />
          </div>
          <button
            onClick={() => onChange(items.filter((_, j) => j !== i))}
            className="mt-3 rounded-full p-1.5 text-white/40 transition hover:text-rose-300"
            aria-label={`Remove ${item[f.main]}`}
          >
            ✕
          </button>
        </div>
      ))}
    </>
  );
}

function EditableRow({ label, value, sub, onChange }: { label: string; value: string | null; sub?: string; onChange: (v: string | null) => void }) {
  const [editing, setEditing] = useState(false);
  if (editing) {
    return (
      <form
        className="flex items-center gap-2 px-4 py-2.5"
        onSubmit={(e) => {
          e.preventDefault();
          const v = (new FormData(e.currentTarget).get("v") as string).trim();
          onChange(v || null);
          setEditing(false);
        }}
      >
        <input
          name="v"
          autoFocus
          defaultValue={value ?? ""}
          aria-label={label}
          className="min-w-0 flex-1 rounded-lg bg-white/10 px-3 py-2 text-sm text-white outline-none ring-1 ring-juno-400"
        />
        <button className="rounded-lg bg-juno-400 px-3 py-2 text-xs font-bold text-night">Save</button>
      </form>
    );
  }
  return (
    <button onClick={() => setEditing(true)} className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition hover:bg-white/[0.03]">
      <span className="min-w-0">
        <span className="block text-[11px] text-white/45">{label}</span>
        <span className={`block truncate text-[15px] ${value ? "text-white" : "text-white/35 italic"}`}>{value ?? "Not given"}</span>
        {sub && <span className="block truncate text-xs text-white/50">{sub}</span>}
      </span>
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0 text-white/30" aria-hidden>
        <path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
      </svg>
    </button>
  );
}
