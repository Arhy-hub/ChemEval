"use client";

// Voice in/out using the browser's Web Speech API.
// Note: in Chrome, speech recognition audio is processed by Google's servers.
import { useCallback, useEffect, useRef, useState } from "react";

/* ---------- Text-to-speech ---------- */

const PREFERRED_VOICES = ["Google UK English Female", "Serena", "Kate", "Martha", "Stephanie", "Microsoft Sonia", "Microsoft Libby"];

function pickVoice(): SpeechSynthesisVoice | null {
  const voices = window.speechSynthesis?.getVoices() ?? [];
  for (const name of PREFERRED_VOICES) {
    const v = voices.find((x) => x.name.includes(name));
    if (v) return v;
  }
  return voices.find((v) => v.lang === "en-GB") ?? voices.find((v) => v.lang.startsWith("en")) ?? null;
}

export function useSpeaker() {
  const [speaking, setSpeaking] = useState(false);
  const [muted, setMuted] = useState(false);
  const mutedRef = useRef(false);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;
    const load = () => (voiceRef.current = pickVoice());
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, []);

  const stop = useCallback(() => {
    window.speechSynthesis?.cancel();
    setSpeaking(false);
  }, []);

  /** Speak text; resolves when finished (or immediately if muted/unsupported). */
  const speak = useCallback((text: string) => {
    return new Promise<void>((resolve) => {
      if (mutedRef.current || !("speechSynthesis" in window)) return resolve();
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text.replace(/[\u{1F300}-\u{1FAFF}☀-➿]/gu, ""));
      if (voiceRef.current) u.voice = voiceRef.current;
      u.lang = voiceRef.current?.lang ?? "en-GB";
      u.rate = 1.04;
      u.pitch = 1.05;
      // Safety net: some browsers never fire onend.
      const timer = setTimeout(done, 1500 + text.length * 90);
      function done() {
        clearTimeout(timer);
        setSpeaking(false);
        resolve();
      }
      u.onstart = () => setSpeaking(true);
      u.onend = done;
      u.onerror = done;
      window.speechSynthesis.speak(u);
    });
  }, []);

  const toggleMute = useCallback(() => {
    mutedRef.current = !mutedRef.current;
    setMuted(mutedRef.current);
    if (mutedRef.current) stop();
  }, [stop]);

  return { speak, stop, speaking, muted, toggleMute };
}

/* ---------- Speech-to-text (hold to talk) ---------- */

type Recognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null;
  onend: (() => void) | null;
  onerror: ((e: { error: string }) => void) | null;
};

function getRecognitionCtor(): (new () => Recognition) | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function useListener() {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<Recognition | null>(null);
  const finalRef = useRef("");
  const interimRef = useRef("");
  const resolveRef = useRef<((text: string) => void) | null>(null);

  useEffect(() => setSupported(!!getRecognitionCtor()), []);

  const start = useCallback(() => {
    const Ctor = getRecognitionCtor();
    if (!Ctor || recRef.current) return;
    const rec = new Ctor();
    rec.lang = "en-GB";
    rec.interimResults = true;
    rec.continuous = true;
    finalRef.current = "";
    interimRef.current = "";
    setInterim("");
    setError(null);
    rec.onresult = (e) => {
      let finals = "";
      let partial = "";
      for (let i = 0; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) finals += r[0].transcript;
        else partial += r[0].transcript;
      }
      finalRef.current = finals;
      interimRef.current = (finals + " " + partial).trim();
      setInterim(interimRef.current);
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") setError("Microphone access is blocked. You can type instead.");
      else if (e.error !== "no-speech" && e.error !== "aborted") setError("I couldn't hear that. Try again or type instead.");
    };
    rec.onend = () => {
      recRef.current = null;
      setListening(false);
      // Use the final text, or the last partial words if the final result never arrived.
      const text = finalRef.current.trim() || interimRef.current;
      resolveRef.current?.(text);
      resolveRef.current = null;
    };
    recRef.current = rec;
    setListening(true);
    try {
      rec.start();
    } catch {
      recRef.current = null;
      setListening(false);
    }
  }, []);

  /** Stop listening and resolve with what was heard (includes the last partial words). */
  const stop = useCallback(() => {
    return new Promise<string>((resolve) => {
      const rec = recRef.current;
      if (!rec) return resolve("");
      resolveRef.current = resolve;
      rec.stop();
    });
  }, []);

  const cancel = useCallback(() => {
    resolveRef.current = null;
    recRef.current?.abort();
    setInterim("");
  }, []);

  return { supported, listening, interim, error, start, stop, cancel };
}
