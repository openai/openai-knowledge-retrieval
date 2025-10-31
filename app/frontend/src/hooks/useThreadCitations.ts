import { useCallback, useEffect, useRef, useState } from "react";

import { getKnowledgeThreadCitationsUrl } from "../lib/config";

export type CitationRecord = {
  document_id: string;
  filename: string;
  title: string;
  description: string | null;
  annotation_index: number | null;
};

type CitationsResponse = {
  documentIds: string[];
  citations: CitationRecord[];
};

export function useThreadCitations(threadId: string | null) {
  const [activeDocumentIds, setActiveDocumentIds] = useState<Set<string>>(new Set());
  const [citations, setCitations] = useState<CitationRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const fetchCitations = useCallback(async () => {
    abortRef.current?.abort();

    if (!threadId) {
      setActiveDocumentIds(new Set());
      setCitations([]);
      setLoading(false);
      setError(null);
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const url = getKnowledgeThreadCitationsUrl(threadId);
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        throw new Error(`Failed to load citations (${response.status})`);
      }

      const payload = (await response.json()) as CitationsResponse;
      setActiveDocumentIds(new Set(payload.documentIds ?? []));
      setCitations(payload.citations ?? []);
    } catch (err) {
      if ((err as DOMException)?.name === "AbortError") {
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setActiveDocumentIds(new Set());
      setCitations([]);
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    }
  }, [threadId]);

  useEffect(() => {
    void fetchCitations();
    return () => {
      abortRef.current?.abort();
    };
  }, [fetchCitations]);

  return {
    citations,
    activeDocumentIds,
    loading,
    error,
    refresh: fetchCitations,
  };
}
