import { useCallback, useState } from "react";
import clsx from "clsx";

import { ChatKitPanel } from "./ChatKitPanel";
import { DocumentPreviewModal } from "./DocumentPreviewModal";
import { KnowledgeDocumentsPanel } from "./KnowledgeDocumentsPanel";
import { ThemeToggle } from "./ThemeToggle";
import type { KnowledgeDocument } from "../hooks/useKnowledgeDocuments";
import { useKnowledgeDocuments } from "../hooks/useKnowledgeDocuments";
import { useThreadCitations } from "../hooks/useThreadCitations";
import type { ColorScheme } from "../hooks/useColorScheme";

type HomeProps = {
  scheme: ColorScheme;
  onThemeChange: (scheme: ColorScheme) => void;
};

export default function Home({ scheme, onThemeChange }: HomeProps) {
  const [selectedDocument, setSelectedDocument] = useState<KnowledgeDocument | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);

  const {
    documents,
    loading: loadingDocuments,
    error: documentsError,
  } = useKnowledgeDocuments();

  const {
    citations,
    activeDocumentIds,
    loading: loadingCitations,
    error: citationsError,
    refresh: refreshCitations,
  } = useThreadCitations(threadId);

  const containerClass = clsx(
    "min-h-screen bg-gradient-to-br transition-colors duration-300",
    scheme === "dark"
      ? "from-slate-950 via-slate-950 to-slate-900 text-slate-100"
      : "from-slate-100 via-white to-slate-200 text-slate-900",
  );

  const handleDocumentSelect = useCallback((document: KnowledgeDocument) => {
    setSelectedDocument(document);
  }, []);

  const handleClosePreview = useCallback(() => {
    setSelectedDocument(null);
  }, []);

  const handleThreadChange = useCallback((nextThreadId: string | null) => {
    setThreadId(nextThreadId);
  }, []);

  const handleResponseCompleted = useCallback(() => {
    void refreshCitations();
  }, [refreshCitations]);

  return (
    <div className={containerClass}>
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-8 px-6 py-8 lg:h-screen lg:max-h-screen lg:py-10">
        <header className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-3">
            <p className="text-sm uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">
              Knowledge Retrieval
            </p>
            <h1 className="text-3xl font-semibold sm:text-4xl">
              Explore your knowledge library
            </h1>
            <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
              Ask questions about the documents you ingested. The assistant runs the retrieval pipeline, grounds the answer in matching snippets, and cites every statement so you can inspect the source.
            </p>
          </div>
          <ThemeToggle value={scheme} onChange={onThemeChange} />
        </header>

        <div className="grid flex-1 grid-cols-1 gap-8 lg:h-[calc(100vh-260px)] lg:grid-cols-[minmax(320px,380px)_1fr] lg:items-stretch xl:grid-cols-[minmax(360px,420px)_1fr]">
          <section className="flex flex-1 flex-col overflow-hidden rounded-3xl bg-white/80 shadow-[0_45px_90px_-45px_rgba(15,23,42,0.6)] ring-1 ring-slate-200/60 backdrop-blur dark:bg-slate-900/70 dark:shadow-[0_45px_90px_-45px_rgba(15,23,42,0.85)] dark:ring-slate-800/60">
            <div className="flex flex-1">
              <ChatKitPanel
                theme={scheme}
                onThreadChange={handleThreadChange}
                onResponseCompleted={handleResponseCompleted}
              />
            </div>
          </section>

          <section className="flex flex-1 flex-col overflow-hidden">
            <KnowledgeDocumentsPanel
              documents={documents}
              activeDocumentIds={activeDocumentIds}
              citations={citations}
              loadingDocuments={loadingDocuments}
              loadingCitations={loadingCitations}
              documentsError={documentsError}
              citationsError={citationsError}
              onSelectDocument={handleDocumentSelect}
            />
          </section>
        </div>
      </div>

      <DocumentPreviewModal document={selectedDocument} onClose={handleClosePreview} />
    </div>
  );
}
