export function DemoBanner() {
  return (
    <div className="flex justify-center pt-[max(0.5rem,env(safe-area-inset-top))]">
      <span className="rounded-full bg-sun/15 px-3 py-1 text-[11px] font-medium text-sun ring-1 ring-sun/25">
        Demo · fake data only, not for real medical use
      </span>
    </div>
  );
}
