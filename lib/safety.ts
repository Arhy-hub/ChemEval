// Client-side safety net: catches obvious emergency phrases before anything is sent to the AI.
// The AI also flags emergencies; this just makes sure the advice appears instantly.
const EMERGENCY = [
  /chest pains?|pain in my chest|heart attack/i,
  /can'?t breathe|cannot breathe|struggling to breathe|can'?t catch my breath/i,
  /\b(kill|hurt|harm)(ing)? myself\b|suicid|end (it all|my life)|don'?t want to (live|be alive)/i,
  /\bstroke\b|face (is )?drooping|slurred speech/i,
  /bleeding (heavily|a lot|won'?t stop)|severe bleeding/i,
  /overdose|collapsed|unconscious|seizure/i,
];

export function looksLikeEmergency(text: string) {
  return EMERGENCY.some((r) => r.test(text));
}

export function mentionsSelfHarm(text: string) {
  return EMERGENCY[2].test(text);
}
