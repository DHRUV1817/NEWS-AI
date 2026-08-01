"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/* N13 · inline ⌘K pill. The affordance is on the surface for newcomers and the
 * shortcut is there for everyone else. Shipping the pill means shipping the
 * keyboard model with it — Esc closes, arrows move, Enter selects, focus is
 * trapped and returned. A div that does none of that is the anti-pattern. */

interface Command {
  group: string;
  label: string;
  hint: string;
  href: string;
}

const COMMANDS: Command[] = [
  { group: "On this page", label: "Run an analysis", hint: "the live tool", href: "#workbench" },
  { group: "On this page", label: "How the pipeline works", hint: "four stages", href: "#pipeline" },
  { group: "On this page", label: "What is measured", hint: "and what is not", href: "#evaluation" },
  { group: "On this page", label: "Architecture", hint: "package layout", href: "#architecture" },
  { group: "Service", label: "OpenAPI schema", hint: "/openapi.json", href: "http://127.0.0.1:8000/openapi.json" },
  { group: "Service", label: "Interactive API docs", hint: "/docs", href: "http://127.0.0.1:8000/docs" },
  { group: "Source", label: "Repository", hint: "github.com/DHRUV1817", href: "https://github.com/DHRUV1817/NEWS-AI" },
];

export default function Nav() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const restoreTo = useRef<HTMLElement | null>(null);

  const matches = COMMANDS.filter((command) =>
    (command.label + command.hint + command.group)
      .toLowerCase()
      .includes(query.trim().toLowerCase()),
  );

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
    setActive(0);
    // Returning focus to the trigger is the half of the keyboard model that
    // gets skipped most often; without it the tab order restarts at the top.
    restoreTo.current?.focus();
  }, []);

  const openPalette = useCallback(() => {
    restoreTo.current = document.activeElement as HTMLElement | null;
    setOpen(true);
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        if (open) close();
        else openPalette();
        return;
      }
      if (!open) return;
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        setActive((i) => (matches.length ? (i + 1) % matches.length : 0));
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActive((i) =>
          matches.length ? (i - 1 + matches.length) % matches.length : 0,
        );
      } else if (event.key === "Enter") {
        const chosen = matches[active];
        if (chosen) {
          event.preventDefault();
          close();
          window.location.href = chosen.href;
        }
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close, openPalette, matches, active]);

  useEffect(() => {
    if (!open) {
      document.body.style.overflow = "";
      return;
    }
    document.body.style.overflow = "hidden";
    const id = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => {
      window.clearTimeout(id);
      document.body.style.overflow = "";
    };
  }, [open]);

  /* Group before rendering rather than tracking the previous group with a
     mutable local — reassigning during render is the kind of thing that works
     until a re-render happens in a different order. */
  const grouped = matches.reduce<
    { group: string; items: { command: Command; index: number }[] }[]
  >((acc, command, index) => {
    const tail = acc[acc.length - 1];
    if (tail && tail.group === command.group) tail.items.push({ command, index });
    else acc.push({ group: command.group, items: [{ command, index }] });
    return acc;
  }, []);

  return (
    <>
      <header className="nav">
        <div className="nav__inner">
          <a className="nav__brand" href="#top">
            NewsNinja
          </a>

          <button
            type="button"
            className="searchpill"
            onClick={openPalette}
            aria-label="Search this page (Command K)"
          >
            <span className="searchpill__text">Search…</span>
            <span className="searchpill__kbd">
              <kbd>⌘</kbd>
              <kbd>K</kbd>
            </span>
          </button>

          <nav className="nav__right">
            <a className="nav__link" href="#evaluation">
              Evaluation
            </a>
            <a
              className="btn btn--ghost"
              href="https://github.com/DHRUV1817/NEWS-AI"
            >
              Source
            </a>
          </nav>
        </div>
      </header>

      <div className={`cmdk${open ? " is-open" : ""}`} aria-hidden={!open}>
        <button
          type="button"
          className="cmdk__backdrop"
          tabIndex={-1}
          aria-label="Close search"
          onClick={close}
        />
        <div className="cmdk__panel" role="dialog" aria-modal="true" aria-label="Search">
          <div className="cmdk__field">
            <input
              ref={inputRef}
              className="cmdk__input"
              value={query}
              placeholder="Search this page…"
              onChange={(event) => {
                setQuery(event.target.value);
                setActive(0);
              }}
            />
            <kbd>esc</kbd>
          </div>

          <div className="cmdk__results">
            {matches.length === 0 && (
              <p className="cmdk__empty">Nothing matches “{query}”.</p>
            )}
            {grouped.map((section) => (
              <div key={section.group}>
                <p className="cmdk__group">{section.group}</p>
                {section.items.map(({ command, index }) => (
                  <a
                    key={command.label}
                    className={`cmdk__item${index === active ? " is-active" : ""}`}
                    href={command.href}
                    onMouseEnter={() => setActive(index)}
                    onClick={close}
                  >
                    <span>{command.label}</span>
                    <span className="cmdk__hint">{command.hint}</span>
                  </a>
                ))}
              </div>
            ))}
          </div>

          <div className="cmdk__foot">
            <span>
              <kbd>↑</kbd>
              <kbd>↓</kbd> navigate
            </span>
            <span>
              <kbd>↵</kbd> open
            </span>
            <span>
              <kbd>esc</kbd> close
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
