"use client";

// Juno's voice-first conversation: sign-up (name + PIN) flows straight into the AI medical interview.
import { useCallback, useEffect, useRef, useState } from "react";
import { Orb } from "./Orb";
import { Keypad, isWeakPin, padError, type PadError } from "./Keypad";
import { ReviewCard } from "./ReviewCard";
import { Captions, Dock, EmergencySheet, TopBar, Transcript, type Msg } from "./voice-ui";
import { useVault } from "@/lib/vault";
import { useListener, useSpeaker } from "@/lib/voice";
import { looksLikeEmergency, mentionsSelfHarm } from "@/lib/safety";
import {
  SECTIONS,
  emptySection,
  mergeSection,
  nextSection,
  type AnySectionData,
  type Profile,
  type SectionId,
  type TurnResult,
} from "@/lib/schema";

type Stage = "welcome" | "name" | "pin1" | "pin2" | "creating" | "interview" | "review" | "complete" | "emergency";
type Turn = { from: "juno" | "user"; text: string };

let n = 0;
const id = () => `m${Date.now()}-${n++}`;

export function Conversation() {
  const { status, profile, createAccount, saveProfile, lock } = useVault();
  const { speak, stop: stopSpeaking, speaking, muted, toggleMute } = useSpeaker();
  const mic = useListener();

  const [stage, setStage] = useState<Stage>(status === "new" ? "welcome" : "interview");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [thinking, setThinking] = useState(false);
  const [retry, setRetry] = useState<(() => void) | null>(null);
  const [showTranscript, setShowTranscript] = useState(false);
  const [padErr, setPadErr] = useState<PadError | null>(null);
  const [section, setSection] = useState<SectionId | null>(null);
  const [review, setReview] = useState<AnySectionData | null>(null);
  const [saving, setSaving] = useState(false);
  const [selfHarm, setSelfHarm] = useState(false);

  // Values the async flow needs without waiting for a re-render.
  const nameRef = useRef(profile?.preferredName ?? "");
  const firstPin = useRef<string | null>(null);
  const profileRef = useRef<Profile | null>(profile);
  const draftRef = useRef<AnySectionData | null>(null);
  const turnsRef = useRef<Turn[]>([]); // this section's turns only — all the AI ever sees
  profileRef.current = profile ?? profileRef.current;

  const say = useCallback(
    async (text: string) => {
      setMessages((m) => [...m, { id: id(), from: "juno", text }]);
      await speak(text);
    },
    [speak],
  );
  const addUser = (text: string) => setMessages((m) => [...m, { id: id(), from: "user", text }]);

  /* ---------- AI interview ---------- */

  const askJuno = useCallback(
    async (sec: SectionId, opts: { resumed?: boolean } = {}) => {
      setRetry(null);
      setThinking(true);
      const basics = profileRef.current?.sections.basics;
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: nameRef.current,
            section: sec,
            draft: draftRef.current,
            messages: turnsRef.current,
            context: sec === "lifestyle" && basics ? { sexAtBirth: basics.sexAtBirth, dateOfBirth: basics.dateOfBirth } : undefined,
            resumed: opts.resumed,
          }),
        });
        if (!res.ok) throw new Error(String(res.status));
        const turn = (await res.json()) as TurnResult;
        draftRef.current = mergeSection(sec, draftRef.current ?? emptySection(sec), turn.extracted);
        turnsRef.current.push({ from: "juno", text: turn.reply });
        setThinking(false);
        if (turn.safety === "emergency") {
          const lastUser = turnsRef.current.filter((t) => t.from === "user").at(-1)?.text ?? "";
          setSelfHarm(mentionsSelfHarm(lastUser));
          setStage("emergency");
        }
        await say(turn.reply);
        if (turn.section_complete && turn.safety !== "emergency") {
          setReview(draftRef.current);
          setStage("review");
        }
      } catch {
        setThinking(false);
        setRetry(() => () => askJuno(sec, opts));
        await say("Sorry, I couldn't connect just then. Tap retry to try again.");
      }
    },
    [say],
  );

  const startSection = useCallback(
    (sec: SectionId, opts: { resumed?: boolean } = {}) => {
      setSection(sec);
      draftRef.current = emptySection(sec);
      turnsRef.current = [];
      setStage("interview");
      return askJuno(sec, opts);
    },
    [askJuno],
  );

  // Coming back after unlocking: pick up where they left off. (New users start from the welcome screen.)
  const started = useRef(status === "new");
  useEffect(() => {
    if (started.current || status !== "unlocked" || stage !== "interview") return;
    started.current = true;
    const p = profileRef.current!;
    nameRef.current = p.preferredName;
    const next = nextSection(p);
    if (next) startSection(next, { resumed: true });
    else {
      setStage("complete");
      say(`Welcome back, ${p.preferredName}! Your profile is all done.`);
    }
  }, [status, stage, startSection, say]);

  /* ---------- Handling what the user says or types ---------- */

  async function handleUser(text: string) {
    text = text.trim();
    if (!text) return;
    stopSpeaking();

    if (stage === "name") {
      const name = cleanName(text);
      addUser(text);
      nameRef.current = name;
      setStage("pin1");
      await say(`Hi ${name}! Can you give me 6 numbers for a login PIN?`);
      return;
    }

    if (stage === "interview" && section) {
      addUser(text);
      if (looksLikeEmergency(text)) {
        // Don't wait for the AI — show help straight away.
        setSelfHarm(mentionsSelfHarm(text));
        setStage("emergency");
        await say("This sounds urgent. Please call 999 or go to A&E now.");
        return;
      }
      turnsRef.current.push({ from: "user", text });
      await askJuno(section);
    }
  }

  async function handlePin(pin: string) {
    if (stage === "pin1") {
      if (isWeakPin(pin)) {
        setPadErr(padError("Too easy to guess. Try another."));
        return;
      }
      firstPin.current = pin;
      setPadErr(null);
      setStage("pin2");
      say("And once more to check.");
      return;
    }
    if (stage === "pin2") {
      if (pin !== firstPin.current) {
        firstPin.current = null;
        setPadErr(padError("Those didn't match"));
        setStage("pin1");
        say("Those didn't match. Let's try again.");
        return;
      }
      setStage("creating");
      addUser("••••••");
      try {
        profileRef.current = await createAccount(nameRef.current, pin);
        firstPin.current = null;
        await say("Ok great! Just so you know, if you forget your PIN I can't get your data back.");
        await startSection(SECTIONS[0].id);
      } catch {
        setStage("pin1");
        setPadErr(padError("Couldn't save. Try again."));
      }
    }
  }

  async function confirmSection(data: AnySectionData) {
    if (!section || !profileRef.current) return;
    setSaving(true);
    try {
      const updated = await saveProfile({
        ...profileRef.current,
        sections: { ...profileRef.current.sections, [section]: data },
      });
      profileRef.current = updated;
      setReview(null);
      const next = nextSection(updated);
      if (next) await startSection(next);
      else {
        setSection(null);
        setStage("complete");
        await say(`That's everything, ${nameRef.current}! Your health profile is ready to share with trials.`);
      }
    } catch {
      await say("Sorry, I couldn't save that on your phone. Please try again.");
    } finally {
      setSaving(false);
    }
  }

  async function fixSection() {
    setReview(null);
    setStage("interview");
    const q = "No problem. What needs changing?";
    turnsRef.current.push({ from: "juno", text: q });
    await say(q);
  }

  /* ---------- Mic ---------- */

  const micDown = () => {
    stopSpeaking();
    mic.start();
  };
  const micUp = async () => {
    const heard = await mic.stop();
    if (heard) handleUser(heard);
  };

  /* ---------- Render ---------- */

  // Caption: Juno's latest line, plus what the user said just before it (or just now, while Juno thinks).
  const last = messages.at(-1) ?? null;
  const lastJunoIdx = messages.findLastIndex((m) => m.from === "juno");
  const lastJuno = lastJunoIdx >= 0 ? messages[lastJunoIdx] : null;
  const lastUser = last?.from === "user" ? last : messages[lastJunoIdx - 1]?.from === "user" ? messages[lastJunoIdx - 1] : null;
  const orbState = mic.listening ? "listening" : thinking || stage === "creating" ? "thinking" : speaking ? "speaking" : "idle";
  const done = SECTIONS.filter((s) => profileRef.current?.sections[s.id]).map((s) => s.id);
  const showProgress = ["interview", "review", "complete", "emergency"].includes(stage) && !!profileRef.current;
  const canTalk = stage === "name" || stage === "interview";
  const compact = ["pin1", "pin2", "creating", "review", "emergency"].includes(stage); // a sheet is open

  if (stage === "welcome") {
    return (
      <div className="flex flex-1 flex-col items-center justify-between px-6 pt-16 pb-[max(2rem,env(safe-area-inset-bottom))]">
        <div className="flex flex-col items-center text-center">
          <Orb state="idle" size={180} />
          <h1 className="animate-rise mt-12 text-3xl font-semibold tracking-tight">Hi, I&apos;m Juno</h1>
          <p className="animate-rise mt-3 max-w-xs text-white/60" style={{ animationDelay: "80ms" }}>
            Build your health profile for clinical trials, just by talking.
          </p>
        </div>
        <div className="w-full">
          <p className="mb-4 flex items-center justify-center gap-1.5 text-xs text-white/45">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><rect x="5" y="11" width="14" height="10" rx="2.5" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>
            Everything stays encrypted on this phone
          </p>
          <button
            onClick={() => {
              setStage("name");
              say("Hi! I'm Juno. What's your name?");
            }}
            className="w-full rounded-full bg-white py-4 text-[17px] font-semibold text-night transition active:scale-[0.98]"
          >
            Start talking
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <TopBar
        progress={showProgress ? { done, current: section } : null}
        muted={muted}
        onMute={toggleMute}
        onTranscript={() => setShowTranscript(true)}
        onLock={profileRef.current ? () => { stopSpeaking(); lock(); } : undefined}
      />

      <div className={`flex min-h-0 flex-1 flex-col items-center justify-center ${compact ? "gap-5 overflow-hidden" : "gap-10"}`}>
        <Orb state={orbState} size={compact ? 76 : 190} />
        <Captions juno={lastJuno} user={lastUser} interim={mic.listening ? mic.interim : ""} thinking={thinking} />
        {retry && !thinking && (
          <button onClick={() => retry()} className="animate-fade -mt-6 rounded-full bg-white/10 px-5 py-2 text-sm font-semibold ring-1 ring-white/15">
            Retry
          </button>
        )}
        {stage === "complete" && (
          <p className="animate-fade -mt-4 rounded-full bg-juno-400/15 px-4 py-1.5 text-xs text-juno-200">
            Your Health Card is coming in the next step
          </p>
        )}
      </div>

      {canTalk && (
        <Dock
          disabled={thinking}
          micSupported={mic.supported}
          listening={mic.listening}
          micError={mic.error}
          placeholder={stage === "name" ? "Your name" : "Type your answer…"}
          onSend={handleUser}
          onMicDown={micDown}
          onMicUp={micUp}
        />
      )}
      {(stage === "pin1" || stage === "pin2" || stage === "creating") && (
        <Keypad
          title={stage === "pin1" ? "Create a 6-digit PIN" : "Confirm your PIN"}
          hint="Only stored on this phone. Juno never sees or hears it."
          error={padErr}
          busy={stage === "creating"}
          onComplete={handlePin}
        />
      )}
      {stage === "review" && review && section && (
        <ReviewCard key={section} section={section} data={review} saving={saving} onConfirm={confirmSection} onFix={fixSection} />
      )}
      {stage === "emergency" && (
        <EmergencySheet
          selfHarm={selfHarm}
          onDismiss={() => {
            setStage("interview");
            say("Ok. Let's carry on whenever you're ready.");
          }}
        />
      )}
      {showTranscript && <Transcript messages={messages} onClose={() => setShowTranscript(false)} />}
    </div>
  );
}

/** "my name is bob." → "Bob" */
function cleanName(text: string) {
  const stripped = text
    .replace(/^(hi|hello|hey)[,!.\s]+/i, "")
    .replace(/^(my name is|my name's|i'm|i am|im|it's|its|call me|this is|you can call me)\s+/i, "")
    .replace(/[.!?,]+$/g, "")
    .trim();
  const words = stripped.split(/\s+/).slice(0, 3).join(" ").slice(0, 40) || text.slice(0, 40);
  return words.replace(/\b\p{L}/gu, (c) => c.toUpperCase());
}
