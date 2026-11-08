export type OrbState = "idle" | "listening" | "thinking" | "speaking";

/** Juno's animated presence. Original design: a glowing teal sphere with swirling light. */
export function Orb({ state, size = 200 }: { state: OrbState; size?: number }) {
  return (
    <div className="relative shrink-0 transition-all duration-500" style={{ width: size, height: size }} aria-hidden>
      {state === "listening" && (
        <>
          <span className="orb-ring" />
          <span className="orb-ring" style={{ animationDelay: "0.8s" }} />
        </>
      )}
      <div className="orb h-full w-full" data-state={state} />
    </div>
  );
}
