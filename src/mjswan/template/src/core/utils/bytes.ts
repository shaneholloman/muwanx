/** Asset bytes: already in hand, or a lazy loader fetched on demand. */
export type Bytes = ArrayBuffer | (() => Promise<ArrayBuffer>);

/** Resolve a {@link Bytes} value to a concrete ArrayBuffer. */
export function resolveBytes(bytes: Bytes): Promise<ArrayBuffer> {
  return typeof bytes === 'function' ? bytes() : Promise.resolve(bytes);
}
