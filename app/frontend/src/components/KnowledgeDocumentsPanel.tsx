import clsx from "clsx";

import type { KnowledgeDocument } from "../hooks/useKnowledgeDocuments";
import type { CitationRecord } from "../hooks/useThreadCitations";

type KnowledgeDocumentsPanelProps = {
  documents: KnowledgeDocument[];
  activeDocumentIds: Set<string>;
  citations: CitationRecord[];
  loadingDocuments: boolean;
  loadingCitations: boolean;
  documentsError: string | null;
  citationsError: string | null;
  onSelectDocument: (document: KnowledgeDocument) => void;
};

export function KnowledgeDocumentsPanel({
  documents,
  activeDocumentIds,
  citations,
  loadingDocuments,
  loadingCitations,
  documentsError,
  citationsError,
  onSelectDocument,
}: KnowledgeDocumentsPanelProps) {
  const statusMessage = getStatusMessage({
    loadingCitations,
    citationsError,
    activeCount: activeDocumentIds.size,
  });

  return (
    <div className="flex h-full flex-col rounded-3xl border border-slate-200/60 bg-white/80 shadow-[0_35px_90px_-45px_rgba(15,23,42,0.55)] ring-1 ring-slate-200/60 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70 dark:shadow-[0_45px_95px_-50px_rgba(15,23,42,0.85)] dark:ring-slate-800/60">
      <div className="border-b border-slate-200/60 px-6 py-5 dark:border-slate-800/60">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              Knowledge library
            </h2>
            <p className="mt-2 max-w-2xl text-sm text-slate-600 dark:text-slate-300">
              Browse your configured knowledge sources. Documents cited in the latest assistant response are highlighted.
            </p>
          </div>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium uppercase tracking-wide text-slate-600 dark:bg-slate-800/70 dark:text-slate-300">
            {loadingDocuments ? "Loading…" : `${documents.length} files`}
          </span>
        </div>
        <p className="mt-4 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {statusMessage}
        </p>
      </div>

      <div className="relative flex-1 overflow-hidden">
        {documentsError ? (
          <ErrorState message={documentsError} />
        ) : (
          <DocumentGrid
            documents={documents}
            loading={loadingDocuments}
            activeDocumentIds={activeDocumentIds}
            onSelectDocument={onSelectDocument}
          />
        )}
      </div>

      {citations.length > 0 ? (
        <aside className="border-t border-slate-200/60 bg-slate-50/60 px-6 py-4 text-sm text-slate-600 dark:border-slate-800/60 dark:bg-slate-900/80 dark:text-slate-300">
          <p className="font-medium text-slate-700 dark:text-slate-200">Latest sources</p>
          <ul className="mt-2 space-y-1">
            {citations.map((citation) => (
              <li
                key={`${citation.document_id}-${citation.annotation_index ?? "na"}`}
                className="flex flex-wrap items-baseline gap-2"
              >
                <span className="font-medium text-slate-700 dark:text-slate-100">
                  {citation.title}
                </span>
                <span className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  {citation.filename}
                </span>
              </li>
            ))}
          </ul>
        </aside>
      ) : null}
    </div>
  );
}

function getStatusMessage({
  loadingCitations,
  citationsError,
  activeCount,
}: {
  loadingCitations: boolean;
  citationsError: string | null;
  activeCount: number;
}) {
  if (loadingCitations) {
    return "Updating citations…";
  }
  if (citationsError) {
    return citationsError;
  }
  if (activeCount > 0) {
    return `${activeCount} source${activeCount === 1 ? "" : "s"} cited in the latest response.`;
  }
  return "No sources cited yet.";
}

function DocumentGrid({
  documents,
  loading,
  activeDocumentIds,
  onSelectDocument,
}: {
  documents: KnowledgeDocument[];
  loading: boolean;
  activeDocumentIds: Set<string>;
  onSelectDocument: (document: KnowledgeDocument) => void;
}) {
  if (loading) {
    return (
      <div className="grid h-full place-items-center bg-gradient-to-br from-slate-50/70 via-white to-slate-100/80 text-slate-500 dark:from-slate-900/40 dark:via-slate-900/20 dark:to-slate-900/60">
        <span className="text-sm font-medium">Loading documents…</span>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="grid h-full place-items-center bg-slate-50/50 text-sm text-slate-500 dark:bg-slate-900/40 dark:text-slate-400">
        No documents available.
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-6">
      <div
        className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3"
        style={{ gridAutoRows: "1fr" }}
      >
        {documents.map((document) => {
          const active = activeDocumentIds.has(document.id);
          const fileVariant = getFileVariant(document.filename);
          return (
            <button
              type="button"
              key={document.id}
              className={clsx(
                "group flex h-full min-h-[260px] flex-col justify-between overflow-hidden rounded-2xl border bg-white/80 p-4 text-left shadow-sm transition-all duration-200 hover:-translate-y-1 hover:shadow-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500 dark:border-slate-800/70 dark:bg-slate-900/80",
                active
                  ? "border-blue-500/70 ring-2 ring-blue-400/60"
                  : "border-slate-200/70 dark:ring-0",
              )}
              onClick={() => onSelectDocument(document)}
            >
              <div className="flex flex-col gap-4">
                <div className="space-y-1">
                  <span
                    className={clsx(
                      "inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
                      fileVariant.badge,
                    )}
                  >
                    {fileVariant.label}
                  </span>
                  <p
                    className="text-xs font-medium text-slate-500 dark:text-slate-400 break-words"
                  >
                    {document.filename}
                  </p>
                </div>
                <div className="flex flex-1 flex-col space-y-2">
                  <h3 className="text-base font-semibold leading-snug text-slate-800 transition-colors group-hover:text-blue-600 dark:text-slate-100 dark:group-hover:text-blue-300 break-words">
                    {document.title}
                  </h3>
                  {document.description ? (
                    <p
                      className="text-sm leading-snug text-slate-600 dark:text-slate-300 line-clamp-3 break-words"
                      style={{
                        display: "-webkit-box",
                        WebkitBoxOrient: "vertical",
                        WebkitLineClamp: 3,
                        overflow: "hidden",
                      }}
                    >
                      {document.description}
                    </p>
                  ) : null}
                </div>
              </div>
              <span
                className={clsx(
                  "mt-6 inline-flex w-fit items-center self-start rounded-full px-3 py-1 text-xs font-medium",
                  active
                    ? "bg-blue-100 text-blue-700 dark:bg-blue-900/60 dark:text-blue-200"
                    : "bg-slate-100 text-slate-600 dark:bg-slate-800/60 dark:text-slate-300",
                )}
              >
                {active ? "Cited in latest response" : "Not yet cited"}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 bg-red-50/70 px-6 text-center text-sm text-red-600 dark:bg-red-950/40 dark:text-red-300">
      <span className="font-semibold">Unable to load documents</span>
      <span>{message}</span>
    </div>
  );
}

type FileVariant = "pdf" | "html" | "default";

function getFileVariant(filename: string): {
  variant: FileVariant;
  label: string;
  badge: string;
} {
  const lower = filename.toLowerCase();
  let variant: FileVariant = "default";
  if (lower.endsWith(".pdf")) variant = "pdf";
  else if (lower.endsWith(".html")) variant = "html";

  const styles: Record<FileVariant, { label: string; badge: string }> = {
    pdf: {
      label: "PDF",
      badge: "bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-200",
    },
    html: {
      label: "HTML",
      badge: "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-200",
    },
    default: {
      label: "FILE",
      badge: "bg-slate-100 text-slate-600 dark:bg-slate-800/60 dark:text-slate-200",
    },
  };

  const style = styles[variant];
  return { variant, ...style };
}
