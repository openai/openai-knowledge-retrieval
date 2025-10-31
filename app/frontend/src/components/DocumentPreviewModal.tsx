import clsx from "clsx";

import type { KnowledgeDocument } from "../hooks/useKnowledgeDocuments";
import { KNOWLEDGE_DOCUMENT_FILE_URL } from "../lib/config";

type DocumentPreviewModalProps = {
  document: KnowledgeDocument | null;
  onClose: () => void;
};

export function DocumentPreviewModal({ document, onClose }: DocumentPreviewModalProps) {
  if (!document) {
    return null;
  }

  const previewUrl = KNOWLEDGE_DOCUMENT_FILE_URL(document.id);
  const fileType = inferFileType(document.filename);

  return (
    <div className="fixed inset-0 z-[999] flex items-center justify-center bg-slate-950/70 px-4 py-10 backdrop-blur-sm">
      <div className="relative flex h-full w-full max-w-5xl flex-col overflow-hidden rounded-3xl border border-slate-200/40 bg-white shadow-2xl ring-1 ring-slate-200/40 dark:border-slate-700/60 dark:bg-slate-950/90 dark:ring-slate-800/60">
        <header className="flex items-start justify-between gap-4 border-b border-slate-200/50 bg-white/90 px-6 py-4 dark:border-slate-800/60 dark:bg-slate-900/80">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
              {fileType === "pdf"
                ? "PDF document"
                : fileType === "html"
                  ? "HTML document"
                  : "File preview"}
            </p>
            <h3 className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100">
              {document.title}
            </h3>
            {document.description ? (
              <p className="mt-2 max-w-3xl text-sm text-slate-600 dark:text-slate-300">
                {document.description}
              </p>
            ) : null}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-slate-200/80 bg-white px-3 py-1 text-sm font-medium text-slate-700 shadow-sm transition hover:-translate-y-0.5 hover:bg-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500 dark:border-slate-700/70 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            Close
          </button>
        </header>

        <div className="flex-1 bg-slate-950/5 dark:bg-slate-900/40">
          <iframe
            key={document.id}
            title={document.title}
            src={previewUrl}
            className={clsx(
              "h-full w-full border-0",
              fileType === "pdf" ? "bg-slate-900/5" : "bg-white",
            )}
            allow="fullscreen"
          />
        </div>
      </div>
    </div>
  );
}

function inferFileType(filename: string) {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".pdf")) return "pdf" as const;
  if (lower.endsWith(".html")) return "html" as const;
  return "file" as const;
}
