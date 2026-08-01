"use client";

import { useEffect } from "react";

/* One IntersectionObserver for the whole page. Anything with .reveal fades and
 * rises once, then keeps its state — re-animating on scroll-back is motion for
 * its own sake. Reduced-motion never arms the observer at all, so those users
 * get the content visible from the first paint rather than a transition that
 * was merely shortened. */
export default function Reveal() {
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const nodes = Array.from(document.querySelectorAll<HTMLElement>(".reveal"));

    if (reduced.matches) {
      nodes.forEach((node) => node.classList.add("is-in"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-in");
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -10% 0px", threshold: 0.05 },
    );

    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  return null;
}
