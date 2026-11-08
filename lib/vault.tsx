"use client";

// Holds the unlocked profile + key in memory only. Locking (or closing the tab) forgets both.
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { PBKDF2_ITERATIONS, decryptJSON, deriveKey, encryptJSON, randomBytes, type Bytes } from "./crypto";
import { getVault, putVault, wipeAll } from "./db";
import { newProfile, type Profile } from "./schema";

type Status = "loading" | "new" | "locked" | "unlocked";

type VaultContextValue = {
  status: Status;
  profile: Profile | null;
  createAccount: (preferredName: string, passcode: string) => Promise<Profile>;
  unlock: (passcode: string) => Promise<boolean>;
  lock: () => void;
  saveProfile: (profile: Profile) => Promise<Profile>;
  wipe: () => Promise<void>;
};

const VaultContext = createContext<VaultContextValue | null>(null);

export function VaultProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<Status>("loading");
  const [profile, setProfile] = useState<Profile | null>(null);
  // Never put these in React state that could be serialised or logged.
  const keyRef = useRef<CryptoKey | null>(null);
  const saltRef = useRef<{ salt: Bytes; iterations: number; createdAt: string } | null>(null);

  useEffect(() => {
    getVault()
      .then((v) => setStatus(v ? "locked" : "new"))
      .catch(() => setStatus("new"));
  }, []);

  const write = useCallback(async (p: Profile) => {
    if (!keyRef.current || !saltRef.current) throw new Error("Vault is locked");
    const { iv, ciphertext } = await encryptJSON(keyRef.current, p);
    await putVault({ version: 1, ...saltRef.current, iv, ciphertext });
  }, []);

  const createAccount = useCallback(
    async (preferredName: string, passcode: string) => {
      const salt = randomBytes(16);
      keyRef.current = await deriveKey(passcode, salt, PBKDF2_ITERATIONS);
      saltRef.current = { salt, iterations: PBKDF2_ITERATIONS, createdAt: new Date().toISOString() };
      const p = newProfile(preferredName);
      await write(p);
      setProfile(p);
      setStatus("unlocked");
      return p;
    },
    [write],
  );

  const unlock = useCallback(async (passcode: string) => {
    const v = await getVault();
    if (!v) {
      setStatus("new");
      return false;
    }
    try {
      const key = await deriveKey(passcode, v.salt, v.iterations);
      const p = await decryptJSON<Profile>(key, v.iv, v.ciphertext);
      p.sections ??= {}; // profiles created before the interview existed
      keyRef.current = key;
      saltRef.current = { salt: v.salt, iterations: v.iterations, createdAt: v.createdAt };
      setProfile(p);
      setStatus("unlocked");
      return true;
    } catch {
      return false; // wrong passcode → AES-GCM auth fails
    }
  }, []);

  const lock = useCallback(() => {
    keyRef.current = null;
    saltRef.current = null;
    setProfile(null);
    setStatus("locked");
  }, []);

  const saveProfile = useCallback(
    async (p: Profile) => {
      const updated = { ...p, updatedAt: new Date().toISOString() };
      await write(updated); // only update memory once it's safely on disk
      setProfile(updated);
      return updated;
    },
    [write],
  );

  const wipe = useCallback(async () => {
    await wipeAll();
    keyRef.current = null;
    saltRef.current = null;
    setProfile(null);
    setStatus("new");
  }, []);

  return (
    <VaultContext.Provider value={{ status, profile, createAccount, unlock, lock, saveProfile, wipe }}>
      {children}
    </VaultContext.Provider>
  );
}

export function useVault() {
  const ctx = useContext(VaultContext);
  if (!ctx) throw new Error("useVault must be used inside <VaultProvider>");
  return ctx;
}
