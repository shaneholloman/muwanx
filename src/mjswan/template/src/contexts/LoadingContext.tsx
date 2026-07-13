import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface LoadingContextType {
  isLoading: boolean;
  loadingMessage: string | null;
  showLoading: (message?: string | null) => void;
  setLoadingMessage: (message: string | null) => void;
  hideLoading: () => void;
}

const LoadingContext = createContext<LoadingContextType | undefined>(undefined);

export const useLoading = () => {
  const context = useContext(LoadingContext);
  if (!context) {
    throw new Error('useLoading must be used within a LoadingProvider');
  }
  return context;
};

interface LoadingProviderProps {
  children: ReactNode;
}

export const LoadingProvider: React.FC<LoadingProviderProps> = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessageState] = useState<string | null>(null);

  // Set-based (not ref-counted): callers map an authoritative loading boolean
  // (chiefly the engine's `state.loading`) to show/hide. Ref-counting assumed
  // balanced show/hide pairs, which the createEngine client does not do —
  // config-load, engine-create, and the engine-state mirror each `showLoading`,
  // so a counter never returned to zero and the overlay stuck after load.
  const showLoading = useCallback((message?: string | null) => {
    if (message !== undefined) {
      setLoadingMessageState(message);
    }
    setIsLoading(true);
  }, []);

  const setLoadingMessage = useCallback((message: string | null) => {
    setLoadingMessageState(message);
  }, []);

  const hideLoading = useCallback(() => {
    setIsLoading(false);
    setLoadingMessageState(null);
  }, []);

  return (
    <LoadingContext.Provider value={{ isLoading, loadingMessage, showLoading, setLoadingMessage, hideLoading }}>
      {children}
    </LoadingContext.Provider>
  );
};
