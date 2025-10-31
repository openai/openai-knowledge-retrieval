import { useCallback, useEffect, useState } from "react";

import { KNOWLEDGE_DOCUMENTS_URL } from "../lib/config";

export type KnowledgeDocument = {
  id: string;
  filename: string;
  title: string;
  description: string | null;
};

type DocumentsResponse = {
  documents: KnowledgeDocument[];
};

export function useKnowledgeDocuments() {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(KNOWLEDGE_DOCUMENTS_URL, {
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        throw new Error(`Failed to load documents (${response.status})`);
      }

      const payload = (await response.json()) as DocumentsResponse;
      setDocuments(payload.documents ?? []);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDocuments();
  }, [fetchDocuments]);

  return { documents, loading, error, refresh: fetchDocuments };
}
