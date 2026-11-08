"use client";

import { useVault } from "@/lib/vault";
import { Conversation } from "@/components/Conversation";
import { LockScreen } from "@/components/LockScreen";
import { Orb } from "@/components/Orb";

export default function Page() {
  const { status } = useVault();

  if (status === "loading") {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Orb state="thinking" size={90} />
      </div>
    );
  }
  if (status === "locked") return <LockScreen />;
  // Same element for "new" and "unlocked", so the conversation carries on after the account is created.
  return <Conversation key="conversation" />;
}
