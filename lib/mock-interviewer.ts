// Scripted stand-in for Claude, used when ANTHROPIC_API_KEY isn't set.
// Lets the app be demoed/tested offline. It is deliberately simple.
import { emptySection, isListSection, type AnySectionData, type SectionId, type TurnResult } from "./schema";

const FIELD_Q: Record<string, string> = {
  fullName: "What's your full name?",
  dateOfBirth: "What's your date of birth?",
  sexAtBirth: "What sex were you assigned at birth? Trials need this for safety.",
  height: "How tall are you?",
  weight: "Roughly how much do you weigh?",
  bloodType: "Do you know your blood type?",
  phone: "What's the best phone number for you?",
  emergencyContactName: "Who should we contact in an emergency?",
  emergencyContactPhone: "And what's their phone number?",
  gpSurgery: "Which GP surgery are you registered with?",
  smoking: "Do you smoke, or have you in the past?",
  alcohol: "Roughly how much alcohol do you drink in a week?",
  exercise: "How active are you most weeks?",
};

const LIST_Q: Record<string, [string, string]> = {
  conditions: ["Do you have any medical conditions, now or in the past?", "conditions"],
  medications: ["Do you take any medicines, including vitamins or inhalers?", "medicines"],
  allergies: ["Are you allergic to anything?", "allergies"],
  procedures: ["Have you had any operations or hospital stays?", "operations"],
  family: ["Do any close relatives have health conditions, like heart disease or diabetes?", "family conditions"],
};

const NO = /^(no|nope|none|nothing|nah|not that i know|i don'?t think so|that'?s (it|all)|no more)\b/i;
const UNSURE = /(don'?t know|not sure|no idea|unsure)/i;

export function mockTurn(section: SectionId, draft: AnySectionData | null, lastUser: string | null): TurnResult {
  const d = (draft ?? emptySection(section)) as Record<string, unknown>;

  if (isListSection(section)) {
    const [firstQ, noun] = LIST_Q[section];
    const items = (d.items as unknown[]) ?? [];
    if (!lastUser) return turn(firstQ, { items: [] });
    if (NO.test(lastUser.trim())) return turn("Ok, thanks! Let me show you what I've got.", { items: [] }, true);
    const item =
      section === "allergies"
        ? { substance: lastUser, reaction: null }
        : section === "family"
          ? { relative: "Family member", condition: lastUser }
          : section === "procedures"
            ? { name: lastUser, when: null, notes: null }
            : section === "medications"
              ? { name: lastUser, dose: null, frequency: null, notes: null }
              : { name: lastUser, status: "unknown", notes: null };
    return turn(items.length ? `Got it. Any other ${noun}?` : `Ok, noted. Any other ${noun}?`, { items: [item] });
  }

  const fields = Object.keys(FIELD_Q).filter((k) => k in d);
  const pending = fields.filter((k) => d[k] == null);
  const extracted: Record<string, string | null> = Object.fromEntries(Object.keys(d).map((k) => [k, null]));
  if (lastUser && pending[0]) {
    extracted[pending[0]] = UNSURE.test(lastUser) ? "Not sure" : lastUser;
    pending.shift();
  }
  if (!pending.length) return turn("Thanks! Let me show you what I've got.", extracted, true);
  const prefix = lastUser ? (UNSURE.test(lastUser) ? "That's fine. " : "Ok great! ") : "";
  return turn(prefix + FIELD_Q[pending[0]], extracted);
}

function turn(reply: string, extracted: unknown, complete = false): TurnResult {
  return { reply, extracted: extracted as AnySectionData, section_complete: complete, safety: "none" };
}
