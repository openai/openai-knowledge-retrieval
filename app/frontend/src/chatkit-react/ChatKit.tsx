// react/ChatKit.tsx (in your lib)
import * as React from "react";
import type { ChatKitControl, ToEventHandlerKey } from "./useChatKit";
import type { ChatKitElement, ChatKitElementEventMap } from "@openai/chatkit";

export interface ChatKitProps extends React.HTMLAttributes<ChatKitElement> {
  control: ChatKitControl;
}

declare module "react" {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace JSX {
    interface IntrinsicElements {
      "openai-chatkit": React.DetailedHTMLProps<
        React.HTMLAttributes<ChatKitElement>,
        ChatKitElement
      >;
    }
  }
}

export const ChatKit = React.forwardRef<ChatKitElement, ChatKitProps>(
  function ChatKit({ control, ...htmlProps }, forwardedRef) {
    const ref = React.useRef<ChatKitElement | null>(null);

    React.useLayoutEffect(() => {
      if (!ref.current) return;

      const el = ref.current;

      if (customElements.get("openai-chatkit")) {
        el.setOptions(control.options);
        return;
      }

      let active = true;
      customElements.whenDefined("openai-chatkit").then(() => {
        if (active && ref.current) {
          ref.current.setOptions(control.options);
        }
      });
      return () => {
        active = false;
      };
    }, [control.options]);

    React.useEffect(() => {
      const el = ref.current;
      if (!el) return;

      const events = {
        "chatkit.error": "onError",
        "chatkit.response.end": "onResponseEnd",
        "chatkit.response.start": "onResponseStart",
        "chatkit.log": "onLog",
        "chatkit.thread.change": "onThreadChange",
      } satisfies {
        [K in keyof ChatKitElementEventMap]: ToEventHandlerKey<K>;
      };

      const abortController = new AbortController();

      for (const eventName of Object.keys(
        events,
      ) as (keyof ChatKitElementEventMap)[]) {
        el.addEventListener(
          eventName,
          (e) => {
            const handler = control.handlers[events[eventName]];
            if (typeof handler === "function") {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              handler(e.detail as any);
            }
          },
          { signal: abortController.signal },
        );
      }

      return () => {
        abortController.abort();
      };
    }, [control.handlers]);

    const setRefs = React.useCallback(
      (chatKit: ChatKitElement | null) => {
        ref.current = chatKit;
        control.setInstance(chatKit);
        if (typeof forwardedRef === "function") {
          forwardedRef(chatKit);
        } else if (forwardedRef) {
          forwardedRef.current = chatKit;
        }
      },
      [control, forwardedRef],
    );

    return <openai-chatkit ref={setRefs} {...htmlProps} />;
  },
);
