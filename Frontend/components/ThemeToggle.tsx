"use client";

import { SunMoonIcon } from "lucide-react";

export function ThemeToggle() {
  function toggle() {
    const isDark = document.documentElement.classList.toggle("dark");
    localStorage.setItem("theme", isDark ? "dark" : "light");
  }

  return (
    <button onClick={toggle} className="px-2 py-1 border rounded">
      <SunMoonIcon/>
    </button>
  );
}