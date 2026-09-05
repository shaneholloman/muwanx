import { describe, it, expect } from 'vitest';
import { applyUrlState, type UrlState } from './urlState';

const SELECTED: UrlState = {
  project: 'Demo Two',
  scene: 'Humanoid',
  policy: 'walk',
  panel: true,
  ref: true,
};

describe('applyUrlState', () => {
  it('writes the whole selection', () => {
    expect(applyUrlState('', SELECTED)).toBe('project=Demo+Two&scene=Humanoid&policy=walk');
  });

  it('replaces a stale selection wholesale', () => {
    // Switching project moves project, scene and policy at once; none of the
    // previous project's values may survive into the new URL.
    const search = '?project=Demo+One&scene=Cartpole&policy=balance';
    expect(applyUrlState(search, SELECTED)).toBe('project=Demo+Two&scene=Humanoid&policy=walk');
  });

  it('drops a parameter whose selection is null', () => {
    // Single-project build, and a scene run with no policy.
    const state = { ...SELECTED, project: null, policy: null };
    expect(applyUrlState('?project=Demo+Two&policy=walk', state)).toBe('scene=Humanoid');
  });

  it('leaves parameters it does not own alone', () => {
    const search = '?config=/other/config.json&hands=1';
    expect(applyUrlState(search, SELECTED)).toBe(
      'config=%2Fother%2Fconfig.json&hands=1&project=Demo+Two&scene=Humanoid&policy=walk',
    );
  });

  it('pins the chrome flags only when they are off', () => {
    expect(applyUrlState('', { ...SELECTED, panel: false, ref: false })).toBe(
      'project=Demo+Two&scene=Humanoid&policy=walk&panel=0&ref=0',
    );
    expect(applyUrlState('?panel=0&ref=0', SELECTED)).toBe(
      'project=Demo+Two&scene=Humanoid&policy=walk',
    );
  });
});
