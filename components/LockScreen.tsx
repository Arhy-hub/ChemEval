"use client";

import { useEffect, useState } from "react";
import { Orb } from "./Orb";
import { Keypad, padError, type PadError } from "./Keypad";
import { useVault } from "@/lib/vault";

const MAX_TRIES = 5;
const COOLDOWN_S = 30;

export function LockScreen() {
  const { unlock, wipe } = useVault();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<PadError | null>(null);
  const [fails, setFails] = useState(0);
  const [cooldown, setCooldown] = useState(0);
  const [confirmWipe, setConfirmWipe] = useState(false);

  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  async function handle(pin: string) {
    if (cooldown > 0) return;
    setBusy(true);
    const ok = await unlock(pin);
    setBusy(false);
    if (ok) return;
    const count = fails + 1;
    setFails(count);
    if (count % MAX_TRIES === 0) {
      setCooldown(COOLDOWN_S);
      setError(padError("Too many tries. Please wait."));
    } else {
      setError(padError("Wrong PIN. Try again."));
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-1 flex-col items-center justify-center gap-6 px-8 text-center">
        <Orb state={busy ? "thinking" : "idle"} size={130} />
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
          <p className="mt-1 text-sm text-white/55">Enter your PIN to carry on with Juno.</p>
        </div>
      </div>

      {confirmWipe ? (
        <div className="animate-sheet rounded-t-[28px] border-t border-rose-400/25 bg-night-2/95 px-5 pt-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] backdrop-blur-xl">
          <h2 className="text-lg font-semibold text-rose-200">Forgot your PIN?</h2>
          <p className="mt-1.5 text-sm text-white/70">
            Your profile is only on this phone, locked with your PIN. There&apos;s no copy on a server, so it can&apos;t be
            recovered. The only option is to <strong className="text-white">wipe everything and start again</strong>. This
            can&apos;t be undone.
          </p>
          <div className="mt-5 flex gap-3">
            <button onClick={() => setConfirmWipe(false)} className="flex-1 rounded-full bg-white/[0.08] py-3.5 text-sm font-semibold ring-1 ring-white/10">
              Go back
            </button>
            <button onClick={() => wipe()} className="flex-1 rounded-full bg-rose-500 py-3.5 text-sm font-bold text-white transition active:scale-[0.97]">
              Wipe &amp; start again
            </button>
          </div>
        </div>
      ) : (
        <Keypad
          title={cooldown > 0 ? `Try again in ${cooldown}s` : "Enter your PIN"}
          error={error}
          busy={busy || cooldown > 0}
          onComplete={handle}
          footer={
            <button onClick={() => setConfirmWipe(true)} className="text-sm text-white/50 underline underline-offset-4">
              Forgot PIN?
            </button>
          }
        />
      )}
    </div>
  );
}
