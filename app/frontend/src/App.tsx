import { useCallback } from "react";

import Home from "./components/Home";
import { useColorScheme, type ColorScheme } from "./hooks/useColorScheme";

export default function App() {
  const { scheme, setScheme } = useColorScheme();

  const handleThemeChange = useCallback(
    (value: ColorScheme) => {
      setScheme(value);
    },
    [setScheme],
  );

  return <Home scheme={scheme} onThemeChange={handleThemeChange} />;
}

