"use client";

// Presentational pieces of the voice-first screen.
import { useEffect, useRef, useState } from "react";
import { SECTIONS, type SectionId } from "@/lib/schema";

export type Msg = { id: string; from: "juno" | "user"; text: string };

/* ---------- Top bar ---------- */

export function TopBar({
  progress,
  muted,
  onMute,
  onTranscript,
  onLock,
}: {
  progress: { done: SectionId[]; current: SectionId | null } | null;
  muted: boolean;
  onMute: () => void;
  onTranscript: () => void;
  onLock?: () => void;
}) {
  const current = SECTIONS.find((s) => s.id === progress?.current);
  return (
    <div className="px-4 pt-3">
      <div className="flex items-center justify-between">
        <span className="text-[15px] font-semibold tracking-tight">Juno</span>
        <div className="flex gap-1.5">
          <IconBtn label="Show conversation" onClick={onTranscript}>
            <path d="M4 6h16M4 12h10M4 18h7" />
          </IconBtn>
          <IconBtn label={muted ? "Turn Juno's voice on" : "Mute Juno's voice"} onClick={onMute}>
            <path d="M11 5 6 9H3v6h3l5 4V5Z" />
            {muted ? <path d="m22 9-6 6M16 9l6 6" /> : <path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" />}
          </IconBtn>
          {onLock && (
            <IconBtn label="Lock" onClick={onLock}>
              <rect x="5" y="11" width="14" height="10" rx="2.5" />
              <path d="M8 11V8a4 4 0 0 1 8 0v3" />
            </IconBtn>
          )}
        </div>
      </div>
      {progress && (
        <div className="mt-3 animate-fade">
          <div className="flex gap-1" aria-label={`${progress.done.length} of ${SECTIONS.length} sections done`}>
            {SECTIONS.map((s) => (
              <span
                key={s.id}
                className={`h-1 flex-1 rounded-full transition-colors duration-500 ${
                  progress.done.includes(s.id) ? "bg-juno-400" : s.id === progress.current ? "bg-juno-400/40" : "bg-white/10"
                }`}
              />
            ))}
          </div>
          <p className="mt-1.5 text-[11px] text-white/45">
            {current ? `${current.title} · ${progress.done.length}/${SECTIONS.length} done` : "Profile complete"}
          </p>
        </div>
      )}
    </div>
  );
}

function IconBtn({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className="flex h-9 w-9 items-center justify-center rounded-full bg-white/[0.06] text-white/70 ring-1 ring-white/5 transition hover:text-white active:scale-90"
    >
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
        {children}
      </svg>
    </button>
  );
}

/* ---------- Captions under the orb ---------- */

export function Captions({ juno, user, interim, thinking }: { juno: Msg | null; user: Msg | null; interim: string; thinking: boolean }) {
  return (
    <div className="flex min-h-[110px] w-full flex-col items-center gap-3 px-6 text-center">
      {interim ? (
        <p className="animate-fade text-lg text-white/70 italic">“{interim}”</p>
      ) : (
        user && (
          <p key={user.id} className="animate-rise line-clamp-2 text-sm text-white/40">
            {user.text}
          </p>
        )
      )}
      {thinking ? (
        <div className="flex gap-1.5 pt-3" aria-label="Juno is thinking">
          {[0, 1, 2].map((i) => (
            <span key={i} className="typing-dot h-2 w-2 rounded-full bg-juno-300" style={{ animationDelay: `${i * 0.15}s` }} />
          ))}
        </div>
      ) : (
        juno && (
          <p key={juno.id} className="animate-rise max-w-sm text-[22px] leading-snug font-medium tracking-tight text-white" aria-live="polite">
            {juno.text}
          </p>
        )
      )}
    </div>
  );
}

/* ---------- Bottom dock: type or hold-to-talk ---------- */

export function Dock({
  disabled,
  micSupported,
  listening,
  micError,
  placeholder,
  onSend,
  onMicDown,
  onMicUp,
}: {
  disabled: boolean;
  micSupported: boolean;
  listening: boolean;
  micError: string | null;
  placeholder: string;
  onSend: (text: string) => void;
  onMicDown: () => void;
  onMicUp: () => void;
}) {
  const [text, setText] = useState("");
  const [hint, setHint] = useState<string | null>(null);
  const downAt = useRef(0);

  useEffect(() => {
    if (!hint) return;
    const t = setTimeout(() => setHint(null), 2200);
    return () => clearTimeout(t);
  }, [hint]);

  const submit = () => {
    const t = text.trim();
    if (!t || disabled) return;
    onSend(t);
    setText("");
  };

  const release = () => {
    if (!listening) return;
    if (Date.now() - downAt.current < 300) setHint("Hold the mic while you talk");
    onMicUp();
  };

  return (
    <div className="px-4 pb-[max(1rem,env(safe-area-inset-bottom))]">
      <p className="mb-2 h-4 text-center text-xs text-white/45">
        {micError ?? hint ?? (listening ? "Listening… let go when you're done" : micSupported ? "Hold to talk, or type" : "")}
      </p>
      <div className="flex items-center gap-2.5">
        <form
          className="flex min-w-0 flex-1 items-center rounded-full bg-white/[0.07] pr-1.5 ring-1 ring-white/10 focus-within:ring-juno-400/60"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={disabled}
            maxLength={500}
            placeholder={placeholder}
            aria-label="Type a message"
            className="min-w-0 flex-1 bg-transparent px-5 py-3.5 text-[15px] text-white outline-none placeholder:text-white/35 disabled:opacity-50"
          />
          {text.trim() && (
            <button aria-label="Send" disabled={disabled} className="flex h-9 w-9 items-center justify-center rounded-full bg-white text-night transition active:scale-90">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          )}
        </form>

        {micSupported && (
          <button
            type="button"
            aria-label="Hold to talk"
            disabled={disabled}
            onPointerDown={(e) => {
              e.preventDefault();
              (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
              downAt.current = Date.now();
              onMicDown();
            }}
            onPointerUp={release}
            onPointerCancel={release}
            onContextMenu={(e) => e.preventDefault()}
            className={`relative flex h-14 w-14 shrink-0 touch-none items-center justify-center rounded-full transition-all duration-200 select-none disabled:opacity-40 ${
              listening ? "scale-110 bg-juno-400 text-night shadow-[0_0_32px_rgba(79,199,176,0.7)]" : "bg-white text-night active:scale-95"
            }`}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="2.5" width="6" height="12" rx="3" />
              <path d="M5 11a7 7 0 0 0 14 0M12 18v3.5" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

/* ---------- Full transcript ---------- */

export function Transcript({ messages, onClose }: { messages: Msg[]; onClose: () => void }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => end.current?.scrollIntoView(), []);
  return (
    <div className="animate-fade absolute inset-0 z-30 flex flex-col bg-night/95 backdrop-blur-xl">
      <div className="flex items-center justify-between px-4 py-3">
        <span className="font-semibold">Conversation</span>
        <button onClick={onClose} className="rounded-full bg-white/[0.08] px-4 py-1.5 text-sm font-medium">
          Done
        </button>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto px-4 pb-6">
        {messages.length === 0 && <p className="pt-10 text-center text-sm text-white/40">Nothing yet.</p>}
        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.from === "user" ? "justify-end" : ""}`}>
            <p
              className={`max-w-[80%] rounded-3xl px-4 py-2.5 text-[15px] ${
                m.from === "user" ? "rounded-br-md bg-juno-500/90 text-white" : "rounded-bl-md bg-white/[0.07] text-white/90"
              }`}
            >
              {m.text}
            </p>
          </div>
        ))}
        <div ref={end} />
      </div>
    </div>
  );
}

/* ---------- Emergency ---------- */

export function EmergencySheet({ selfHarm, onDismiss }: { selfHarm: boolean; onDismiss: () => void }) {
  return (
    <div className="animate-sheet rounded-t-[28px] border-t border-rose-400/30 bg-[#2a0f14]/95 px-5 pt-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] backdrop-blur-xl">
      <h2 className="text-lg font-semibold text-rose-100">Please get help now</h2>
      <p className="mt-1 text-sm text-rose-100/75">
        If this is an emergency, call 999 or go to A&amp;E. If it&apos;s urgent but not life-threatening, call NHS 111.
      </p>
      <div className="mt-4 grid grid-cols-2 gap-2.5">
        <a href="tel:999" className="rounded-2xl bg-rose-500 py-3.5 text-center font-bold text-white">Call 999</a>
        <a href="tel:111" className="rounded-2xl bg-white/10 py-3.5 text-center font-semibold text-white">Call NHS 111</a>
        {selfHarm && (
          <a href="tel:116123" className="col-span-2 rounded-2xl bg-white/10 py-3.5 text-center font-semibold text-white">
            Samaritans · 116 123 (free, 24/7)
          </a>
        )}
      </div>
      <button onClick={onDismiss} className="mt-4 w-full py-2 text-sm text-white/55 underline">
        I&apos;m safe. Carry on with the questions
      </button>
    </div>
  );
}
