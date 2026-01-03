import {Grid, GridOptions} from './Grid.js';

/**
 * A layout container designed to hold secondary UI elements, such
 * as an exit button or settings icon. It typically "orbits" or remains
 * attached to a corner of its parent panel, outside the main content area.
 */

export type OrbiterPosition =
  | 'top-right'
  | 'top-left'
  | 'bottom-right'
  | 'bottom-left'
  | 'top'
  | 'bottom'
  | 'left'
  | 'right';

export type OrbiterOptions = GridOptions & {
  orbiterPosition?: OrbiterPosition;
  orbiterScale?: number;
  offset?: number;
  elevation?: number;
};

export class Orbiter extends Grid {
  orbiterPosition: OrbiterPosition;
  orbiterScale: number;
  offset: number;
  elevation: number;

  // These values are based on Material Design guidelines: https://developer.android.com/design/ui/xr/guides/spatial-ui
  private static readonly BASE_OFFSET = 0.02; // put the orbiter outside of the parent panel's "draggable region" by default
  private static readonly BASE_ELEVATION = 0.02; // put the orbiter at 15dp above the parent panel by default
  private static readonly MAX_OUTWARD = 0.05; // avoid the orbiter being too far away from the parent panel

  constructor(options: OrbiterOptions = {}) {
    const {
      orbiterPosition = 'top-left',
      orbiterScale = 0.2,
      offset = 0.0,
      elevation = 0.0,
      ...gridOptions
    } = options;

    super(gridOptions);

    this.orbiterPosition = orbiterPosition;
    this.orbiterScale = orbiterScale;
    this.offset = offset;
    this.elevation = elevation;
  }

  init() {
    super.init();
    this.scale.set(this.orbiterScale, this.orbiterScale, 1.0);
    this._place();
  }

  private _place() {
    const hx = this.rangeX * 0.5;
    const hy = this.rangeY * 0.5;

    const rightEdge = +hx;
    const leftEdge = -hx;
    const topEdge = +hy;
    const bottomEdge = -hy;

    // Clamp edge spacing so the orbiter stays within the recommended range:
    // edgeDelta == -orbiterScale / 2 corresponds to the 50% overlap boundary.
    const edgeDelta = Math.max(
      -this.orbiterScale / 2,
      Math.min(Orbiter.MAX_OUTWARD, Orbiter.BASE_OFFSET + this.offset)
    );

    // Clamp elevation so the orbiter remains in front of the parent panel and doesn’t float excessively.
    const zDelta = Math.max(
      0,
      Math.min(Orbiter.MAX_OUTWARD, Orbiter.BASE_ELEVATION + this.elevation)
    );

    let x = 0.0;
    let y = 0.0;

    switch (this.orbiterPosition) {
      case 'top':
        x = 0.0;
        y = topEdge + this.orbiterScale / 2 + edgeDelta;
        break;
      case 'bottom':
        x = 0.0;
        y = bottomEdge - this.orbiterScale / 2 - edgeDelta;
        break;
      case 'right':
        x = rightEdge + this.orbiterScale / 2 + edgeDelta;
        y = 0.0;
        break;
      case 'left':
        x = leftEdge - this.orbiterScale / 2 - edgeDelta;
        y = 0.0;
        break;
      case 'top-right':
        x = rightEdge - this.orbiterScale / 2;
        y = topEdge + this.orbiterScale / 2 + edgeDelta;
        break;
      case 'top-left':
        x = leftEdge + this.orbiterScale / 2;
        y = topEdge + this.orbiterScale / 2 + edgeDelta;
        break;
      case 'bottom-right':
        x = rightEdge - this.orbiterScale / 2;
        y = bottomEdge - this.orbiterScale / 2 - edgeDelta;
        break;
      case 'bottom-left':
        x = leftEdge + this.orbiterScale / 2;
        y = bottomEdge - this.orbiterScale / 2 - edgeDelta;
        break;
    }

    const z = this.position.z + zDelta;
    this.position.set(x, y, z);
  }
}
