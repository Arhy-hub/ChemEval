"use client";

// Secure 6-digit PIN entry. Digits stay inside this component and go straight to local
// crypto — they are never shown in the conversation, spoken, or sent anywhere.
import { useEffect, useState } from "react";

/** `at` makes a repeated identical error still trigger the shake. */
export type PadError = { msg: string; at: number };
export const padError = (msg: string): PadError => ({ msg, at: Date.now() });

export function Keypad({
  title,
  hint,
  error,
  busy,
  onComplete,
  footer,
}: {
  title: string;
  hint?: string;
  error?: PadError | null;
  busy?: boolean;
  onComplete: (pin: string) => void;
  footer?: React.ReactNode;
}) {
  const [pin, setPin] = useState("");

  useEffect(() => setPin(""), [title, error]);

  const press = (d: string) => {
    if (busy || pin.length >= 6) return;
    const next = pin + d;
    setPin(next);
    if (next.length === 6) setTimeout(() => onComplete(next), 120); // let the last dot fill first
  };
  const back = () => !busy && setPin((cur) => cur.slice(0, -1));

  // Physical keyboard support.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (/^\d$/.test(e.key)) press(e.key);
      else if (e.key === "Backspace") back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="animate-sheet rounded-t-[28px] border-t border-line bg-night-2/95 px-6 pt-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] backdrop-blur-xl">
      <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-white/15" />
      <div className="flex items-center justify-center gap-2 text-sm font-semibold text-white/90">
        <LockIcon /> {title}
      </div>
      {hint && <p className="mt-1 text-center text-xs text-white/45">{hint}</p>}

      <div key={error?.at} className={`my-5 flex justify-center gap-3.5 ${error ? "animate-shake" : ""}`} role="status" aria-label={`${pin.length} of 6 digits entered`}>
        {Array.from({ length: 6 }).map((_, i) => (
          <span
            key={i}
            className={`h-3.5 w-3.5 rounded-full transition-all duration-150 ${
              i < pin.length ? "scale-110 bg-juno-300 shadow-[0_0_12px_rgba(127,220,200,0.8)]" : "bg-white/15"
            }`}
          />
        ))}
      </div>
      <p className="-mt-2 mb-3 h-5 text-center text-sm font-medium text-rose-300" role="alert">
        {busy ? <span className="text-juno-300">Checking…</span> : error?.msg}
      </p>

      <div className="mx-auto grid max-w-[280px] grid-cols-3 gap-3">
        {["1", "2", "3", "4", "5", "6", "7", "8", "9"].map((d) => (
          <Key key={d} onClick={() => press(d)}>{d}</Key>
        ))}
        <span />
        <Key onClick={() => press("0")}>0</Key>
        <button
          type="button"
          onClick={back}
          aria-label="Delete"
          className="flex h-16 items-center justify-center rounded-full text-white/60 transition active:scale-90 active:text-white"
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 5H9l-6 7 6 7h12a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1Z" />
            <path d="m17 9-6 6M11 9l6 6" />
          </svg>
        </button>
      </div>
      {footer && <div className="mt-3 text-center">{footer}</div>}
    </div>
  );
}

function Key({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="h-16 rounded-full bg-white/[0.07] text-2xl font-medium text-white ring-1 ring-white/5 transition active:scale-90 active:bg-white/20"
    >
      {children}
    </button>
  );
}

function LockIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#7fdcc8" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
      <rect x="5" y="11" width="14" height="10" rx="2.5" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </svg>
  );
}

/** Reject PINs that are trivially guessable (all one digit, or straight runs). */
export function isWeakPin(code: string) {
  if (/^(\d)\1{5}$/.test(code)) return true;
  return ["0123456789", "9876543210"].some((r) => r.includes(code));
}
