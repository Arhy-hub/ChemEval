// The patient's medical profile. The whole Profile object is encrypted before it is saved.
// Zod schemas double as the structured-output format Claude must fill in each turn.
import { z } from "zod";

// "Not sure" / "Skipped" are valid answers for any field.
const s = () => z.string().nullable();

export const sectionSchemas = {
  basics: z.object({
    fullName: s(),
    dateOfBirth: s().describe("YYYY-MM-DD if known, otherwise as given"),
    sexAtBirth: s(),
    height: s().describe("with units, e.g. '175 cm' or '5 ft 9 in'"),
    weight: s().describe("with units, e.g. '70 kg' or '11 st 2 lb'"),
    bloodType: s(),
  }),
  contact: z.object({
    phone: s(),
    emergencyContactName: s(),
    emergencyContactPhone: s(),
    gpSurgery: s(),
  }),
  conditions: z.object({
    items: z.array(z.object({ name: z.string(), status: z.enum(["current", "past", "unknown"]), notes: s() })),
  }),
  medications: z.object({
    items: z.array(
      z.object({
        name: z.string().describe("generic name where possible, e.g. 'salbutamol inhaler'"),
        dose: s(),
        frequency: s().describe("e.g. 'twice a day', 'as needed'"),
        notes: s(),
      }),
    ),
  }),
  allergies: z.object({
    items: z.array(z.object({ substance: z.string(), reaction: s() })),
  }),
  procedures: z.object({
    items: z.array(z.object({ name: z.string(), when: s(), notes: s() })),
  }),
  family: z.object({
    items: z.array(z.object({ relative: z.string(), condition: z.string() })),
  }),
  lifestyle: z.object({
    smoking: s(),
    alcohol: s(),
    exercise: s(),
    pregnancy: s().describe("only if relevant; otherwise null"),
  }),
} as const;

export type SectionId = keyof typeof sectionSchemas;
export type SectionData<K extends SectionId> = z.infer<(typeof sectionSchemas)[K]>;
export type AnySectionData = SectionData<SectionId>;

export const SECTIONS: { id: SectionId; title: string; short: string; goal: string }[] = [
  { id: "basics", title: "About you", short: "Basics", goal: "full name, date of birth, sex at birth, height, weight, blood type" },
  { id: "contact", title: "Contact details", short: "Contact", goal: "phone number, an emergency contact (name + phone), their GP surgery" },
  { id: "conditions", title: "Health conditions", short: "Conditions", goal: "current and past health conditions or diagnoses" },
  { id: "medications", title: "Medicines", short: "Medicines", goal: "every medicine they take: name, dose, how often — including over-the-counter, inhalers, creams, vitamins and supplements" },
  { id: "allergies", title: "Allergies", short: "Allergies", goal: "allergies (medicines, foods, other) and what reaction happens" },
  { id: "procedures", title: "Operations & hospital stays", short: "Operations", goal: "operations, surgeries and hospital stays, roughly when" },
  { id: "family", title: "Family health", short: "Family", goal: "health conditions in close blood relatives (parents, siblings, grandparents)" },
  { id: "lifestyle", title: "Lifestyle", short: "Lifestyle", goal: "smoking, alcohol, exercise, and pregnancy or breastfeeding ONLY if relevant (female sex at birth, roughly 16–55)" },
];

export const FIELD_LABELS: Record<string, string> = {
  fullName: "Full name",
  dateOfBirth: "Date of birth",
  sexAtBirth: "Sex at birth",
  height: "Height",
  weight: "Weight",
  bloodType: "Blood type",
  phone: "Phone",
  emergencyContactName: "Emergency contact",
  emergencyContactPhone: "Emergency contact phone",
  gpSurgery: "GP surgery",
  smoking: "Smoking",
  alcohol: "Alcohol",
  exercise: "Exercise",
  pregnancy: "Pregnancy",
};

export const isListSection = (id: SectionId) =>
  id === "conditions" || id === "medications" || id === "allergies" || id === "procedures" || id === "family";

export type Profile = {
  preferredName: string;
  createdAt: string;
  updatedAt: string;
  /** Only sections the user has confirmed ("Looks good") are stored here. */
  sections: Partial<{ [K in SectionId]: SectionData<K> }>;
};

export function newProfile(preferredName: string): Profile {
  const now = new Date().toISOString();
  return { preferredName, createdAt: now, updatedAt: now, sections: {} };
}

export function nextSection(p: Profile): SectionId | null {
  return SECTIONS.find((s) => !p.sections[s.id])?.id ?? null;
}

export function emptySection(id: SectionId): AnySectionData {
  if (isListSection(id)) return { items: [] } as AnySectionData;
  const shape = sectionSchemas[id].shape as Record<string, unknown>;
  return Object.fromEntries(Object.keys(shape).map((k) => [k, null])) as AnySectionData;
}

/** Merge what Claude extracted this turn into the section draft. */
export function mergeSection(id: SectionId, draft: AnySectionData, extracted: AnySectionData | null | undefined): AnySectionData {
  if (!extracted) return draft;
  if (isListSection(id)) {
    const cur = (draft as { items: Record<string, unknown>[] }).items;
    const add = ((extracted as { items?: Record<string, unknown>[] }).items ?? []).filter(Boolean);
    const keyOf = (x: Record<string, unknown>) => String(x.name ?? x.substance ?? `${x.relative}:${x.condition}`).toLowerCase().trim();
    const merged = [...cur];
    for (const item of add) {
      const i = merged.findIndex((m) => keyOf(m) === keyOf(item));
      if (i >= 0) merged[i] = { ...merged[i], ...Object.fromEntries(Object.entries(item).filter(([, v]) => v != null)) };
      else merged.push(item);
    }
    return { items: merged } as AnySectionData;
  }
  const out = { ...(draft as Record<string, unknown>) };
  for (const [k, v] of Object.entries(extracted as Record<string, unknown>)) if (v != null && v !== "") out[k] = v;
  return out as AnySectionData;
}

/** The JSON shape Claude returns every turn (section-specific `extracted`). */
export function turnSchema(id: SectionId) {
  return z.object({
    reply: z.string().describe("What Juno says next. One short sentence or two, max ~20 words, at most ONE question."),
    extracted: sectionSchemas[id].describe("ONLY new or corrected info from the user's latest message; null for anything not mentioned"),
    section_complete: z.boolean().describe("true once every topic in this section is answered, skipped or 'not sure'"),
    safety: z.enum(["none", "emergency", "medical_advice"]),
  });
}
export type TurnResult = { reply: string; extracted: AnySectionData; section_complete: boolean; safety: "none" | "emergency" | "medical_advice" };
