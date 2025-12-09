import * as THREE from 'three';

import {GestureConfiguration} from '../GestureRecognitionOptions';
import {GestureDetectorMap, HandContext} from '../GestureTypes';

type FingerName = 'index' | 'middle' | 'ring' | 'pinky';

const EPSILON = 1e-6;
const FINGER_ORDER: FingerName[] = ['index', 'middle', 'ring', 'pinky'];
const FINGER_PREFIX: Record<FingerName, string> = {
  index: 'index-finger',
  middle: 'middle-finger',
  ring: 'ring-finger',
  pinky: 'pinky-finger',
};

export const heuristicDetectors: GestureDetectorMap = {
  pinch: computePinch,
  'open-palm': computeOpenPalm,
  fist: computeFist,
  'thumbs-up': computeThumbsUp,
  point: computePoint,
  spread: computeSpread,
};

function computePinch(context: HandContext, config: GestureConfiguration) {
  const thumb = getJoint(context, 'thumb-tip');
  const index = getJoint(context, 'index-finger-tip');
  if (!thumb || !index) return undefined;

  const supportMetrics = (['middle', 'ring', 'pinky'] as FingerName[])
    .map((finger) => computeFingerMetric(context, finger))
    .filter(Boolean) as FingerMetrics[];
  const supportCurl =
    supportMetrics.length > 0
      ? average(supportMetrics.map((metrics) => metrics.curlRatio))
      : 1;
  const supportPenalty = clamp01((supportCurl - 1.05) / 0.35);

  const handScale = estimateHandScale(context);
  const threshold = config.threshold ?? Math.max(0.018, handScale * 0.35);
  const distance = thumb.distanceTo(index);

  if (!Number.isFinite(distance) || distance < EPSILON) {
    return {confidence: 0};
  }

  const tightness = clamp01(1 - distance / (threshold * 0.85));
  const loosePenalty = clamp01(1 - distance / (threshold * 1.4));
  let confidence = clamp01(
    distance <= threshold ? tightness : loosePenalty * 0.4
  );
  confidence *= 1 - supportPenalty * 0.45;
  confidence = clamp01(confidence);

  return {
    confidence,
    data: {distance, threshold, supportPenalty},
  };
}

function computeOpenPalm(context: HandContext, config: GestureConfiguration) {
  const fingerMetrics = getFingerMetrics(context);
  if (!fingerMetrics.length) return undefined;
  const handScale = estimateHandScale(context);
  const palmWidth = getPalmWidth(context) ?? handScale * 0.85;
  const palmUp = getPalmUp(context);

  const extensionScores = fingerMetrics.map(({tipDistance}) =>
    clamp01((tipDistance - handScale * 0.5) / (handScale * 0.45))
  );
  const straightnessScores = fingerMetrics.map(({curlRatio}) =>
    clamp01((curlRatio - 1.1) / 0.5)
  );
  const orientationScore =
    palmUp && fingerMetrics.length
      ? average(
          fingerMetrics.map((metrics) =>
            fingerAlignmentScore(context, metrics, palmUp)
          )
        )
      : 0.5;

  const neighbors = getAdjacentFingerDistances(context);
  const spreadScore =
    neighbors.average !== Infinity && palmWidth > EPSILON
      ? clamp01((neighbors.average - palmWidth * 0.55) / (palmWidth * 0.35))
      : 0;

  const extensionScore = average(extensionScores);
  const straightScore = average(straightnessScores);
  const confidence = clamp01(
    extensionScore * 0.4 +
      straightScore * 0.25 +
      spreadScore * 0.2 +
      orientationScore * 0.15
  );

  return {
    confidence,
    data: {
      extensionScore,
      straightScore,
      spreadScore,
      orientationScore,
      threshold: config.threshold,
    },
  };
}

function computeFist(context: HandContext, config: GestureConfiguration) {
  const fingerMetrics = getFingerMetrics(context);
  if (!fingerMetrics.length) return undefined;
  const handScale = estimateHandScale(context);
  const palmWidth = getPalmWidth(context) ?? handScale * 0.85;

  const tipAverage = average(
    fingerMetrics.map((metrics) => metrics.tipDistance)
  );
  const curlAverage = average(
    fingerMetrics.map((metrics) => metrics.curlRatio)
  );

  const neighbors = getAdjacentFingerDistances(context);
  const clusterScore =
    neighbors.average !== Infinity && palmWidth > EPSILON
      ? clamp01((palmWidth * 0.5 - neighbors.average) / (palmWidth * 0.35))
      : 0;
  const thumbTip = getJoint(context, 'thumb-tip');
  const indexBase =
    getFingerJoint(context, 'index', 'phalanx-proximal') ??
    getFingerJoint(context, 'index', 'metacarpal');
  const thumbWrapScore =
    thumbTip && indexBase && palmWidth > EPSILON
      ? clamp01(
          (palmWidth * 0.55 - thumbTip.distanceTo(indexBase)) /
            (palmWidth * 0.35)
        )
      : 0;

  const tipScore = clamp01(
    (handScale * 0.55 - tipAverage) / (handScale * 0.25)
  );
  const curlScore = clamp01((1.08 - curlAverage) / 0.25);
  const confidence = clamp01(
    tipScore * 0.45 +
      curlScore * 0.3 +
      clusterScore * 0.1 +
      thumbWrapScore * 0.15
  );

  return {
    confidence,
    data: {
      tipAverage,
      curlAverage,
      clusterScore,
      thumbWrapScore,
      threshold: config.threshold,
    },
  };
}

function computeThumbsUp(context: HandContext, config: GestureConfiguration) {
  const thumbMetrics = getThumbMetrics(context);
  const fingerMetrics = getFingerMetrics(context);
  if (!thumbMetrics || fingerMetrics.length < 2) return undefined;

  const handScale = estimateHandScale(context);
  const palmWidth = getPalmWidth(context) ?? handScale * 0.85;
  const palmUp = getPalmUp(context);

  const otherCurls = fingerMetrics.map((m) => m.curlRatio);
  const curledScore = clamp01((1.05 - average(otherCurls)) / 0.25);

  const thumbReachRatio =
    thumbMetrics.referenceDistance > EPSILON
      ? thumbMetrics.tipDistance / thumbMetrics.referenceDistance
      : 0;
  const thumbExtendedScore = clamp01((thumbReachRatio - 1.15) / 0.5);

  const indexTip = getJoint(context, 'index-finger-tip');
  const thumbIndexDistance = indexTip
    ? thumbMetrics.tip.distanceTo(indexTip)
    : 0;
  const separationScore =
    palmWidth > EPSILON
      ? clamp01((thumbIndexDistance - palmWidth * 0.4) / (palmWidth * 0.25))
      : 0;

  let orientationScore = 0;
  if (palmUp) {
    const thumbVector = new THREE.Vector3()
      .copy(thumbMetrics.tip)
      .sub(thumbMetrics.metacarpal ?? thumbMetrics.tip);
    if (thumbVector.lengthSq() > EPSILON) {
      thumbVector.normalize();
      const alignment = thumbVector.dot(palmUp);
      orientationScore = clamp01((alignment - 0.35) / 0.35);
    }
  }

  const confidence = clamp01(
    thumbExtendedScore * 0.3 +
      curledScore * 0.35 +
      orientationScore * 0.2 +
      separationScore * 0.15
  );

  return {
    confidence,
    data: {
      thumbReachRatio,
      curledScore,
      orientationScore,
      separationScore,
      threshold: config.threshold,
    },
  };
}

function computePoint(context: HandContext, config: GestureConfiguration) {
  const indexMetrics = computeFingerMetric(context, 'index');
  if (!indexMetrics) return undefined;
  const otherMetrics = FINGER_ORDER.slice(1)
    .map((finger) => computeFingerMetric(context, finger))
    .filter(Boolean) as FingerMetrics[];
  if (!otherMetrics.length) return undefined;

  const handScale = estimateHandScale(context);
  const palmWidth = getPalmWidth(context) ?? handScale * 0.85;
  const palmUp = getPalmUp(context);
  const indexCurlScore = clamp01((indexMetrics.curlRatio - 1.2) / 0.35);
  const indexReachScore = clamp01(
    (indexMetrics.tipDistance - handScale * 0.6) / (handScale * 0.25)
  );
  const indexDirectionScore =
    palmUp && indexMetrics
      ? fingerAlignmentScore(context, indexMetrics, palmUp)
      : 0.4;

  const othersCurl = average(otherMetrics.map((metrics) => metrics.curlRatio));
  const othersCurledScore = clamp01((1.05 - othersCurl) / 0.25);
  const thumbTip = getJoint(context, 'thumb-tip');
  const thumbTuckedScore =
    thumbTip && indexMetrics.metacarpal && palmWidth > EPSILON
      ? clamp01(
          (palmWidth * 0.75 - thumbTip.distanceTo(indexMetrics.metacarpal)) /
            (palmWidth * 0.4)
        )
      : 0.5;

  const confidence = clamp01(
    indexCurlScore * 0.35 +
      indexReachScore * 0.25 +
      othersCurledScore * 0.2 +
      indexDirectionScore * 0.1 +
      thumbTuckedScore * 0.1
  );

  return {
    confidence,
    data: {
      indexCurlScore,
      indexReachScore,
      othersCurledScore,
      indexDirectionScore,
      thumbTuckedScore,
      threshold: config.threshold,
    },
  };
}

function computeSpread(context: HandContext, config: GestureConfiguration) {
  const fingerMetrics = getFingerMetrics(context);
  if (!fingerMetrics.length) return undefined;

  const handScale = estimateHandScale(context);
  const palmWidth = getPalmWidth(context) ?? handScale * 0.85;
  const neighbors = getAdjacentFingerDistances(context);
  const palmUp = getPalmUp(context);

  const spreadScore =
    neighbors.average !== Infinity && palmWidth > EPSILON
      ? clamp01((neighbors.average - palmWidth * 0.6) / (palmWidth * 0.35))
      : 0;
  const extensionScore = clamp01(
    (average(fingerMetrics.map((metrics) => metrics.curlRatio)) - 1.15) / 0.45
  );
  const orientationScore =
    palmUp && fingerMetrics.length
      ? average(
          fingerMetrics.map((metrics) =>
            fingerAlignmentScore(context, metrics, palmUp)
          )
        )
      : 0.5;

  const confidence = clamp01(
    spreadScore * 0.55 + extensionScore * 0.3 + orientationScore * 0.15
  );

  return {
    confidence,
    data: {
      spreadScore,
      extensionScore,
      orientationScore,
      threshold: config.threshold,
    },
  };
}

type FingerMetrics = {
  tip: THREE.Vector3;
  metacarpal?: THREE.Vector3;
  referenceDistance: number;
  tipDistance: number;
  curlRatio: number;
};

type ThumbMetrics = {
  tip: THREE.Vector3;
  metacarpal?: THREE.Vector3;
  referenceDistance: number;
  tipDistance: number;
};

function computeFingerMetric(context: HandContext, finger: FingerName) {
  const tip = getFingerJoint(context, finger, 'tip');
  const proximal = getFingerJoint(context, finger, 'phalanx-proximal');
  const metacarpal = getFingerJoint(context, finger, 'metacarpal');
  const wrist = getJoint(context, 'wrist');
  if (!tip || !wrist) return null;

  const reference = proximal ?? metacarpal;
  if (!reference) return null;

  const referenceDistance = reference.distanceTo(wrist);
  const tipDistance = tip.distanceTo(wrist);
  const curlRatio =
    referenceDistance > EPSILON ? tipDistance / referenceDistance : 0;

  return {
    tip,
    metacarpal,
    referenceDistance,
    tipDistance,
    curlRatio,
  };
}

function getFingerMetrics(context: HandContext) {
  return FINGER_ORDER.map((finger) =>
    computeFingerMetric(context, finger)
  ).filter(Boolean) as FingerMetrics[];
}

function getThumbMetrics(context: HandContext): ThumbMetrics | undefined {
  const tip = getJoint(context, 'thumb-tip');
  const wrist = getJoint(context, 'wrist');
  if (!tip || !wrist) return undefined;

  const metacarpal =
    getJoint(context, 'thumb-metacarpal') ??
    getJoint(context, 'thumb-phalanx-proximal');
  if (!metacarpal) return undefined;

  const referenceDistance = metacarpal.distanceTo(wrist);
  const tipDistance = tip.distanceTo(wrist);

  return {
    tip,
    metacarpal,
    referenceDistance,
    tipDistance,
  };
}

function estimateHandScale(context: HandContext) {
  const wrist = getJoint(context, 'wrist');
  const middleTip = getJoint(context, 'middle-finger-tip');
  const middleBase = getJoint(context, 'middle-finger-metacarpal');
  const palmWidth = getPalmWidth(context);

  const measurements: number[] = [];
  if (wrist && middleTip) measurements.push(middleTip.distanceTo(wrist));
  if (palmWidth) measurements.push(palmWidth);
  if (wrist && middleBase) measurements.push(middleBase.distanceTo(wrist) * 2);

  if (!measurements.length) return 0.08;
  return average(measurements);
}

function getPalmWidth(context: HandContext) {
  const indexBase = getFingerJoint(context, 'index', 'metacarpal');
  const pinkyBase = getFingerJoint(context, 'pinky', 'metacarpal');
  if (!indexBase || !pinkyBase) return null;
  return indexBase.distanceTo(pinkyBase);
}

function getPalmNormal(context: HandContext) {
  const wrist = getJoint(context, 'wrist');
  const indexBase = getFingerJoint(context, 'index', 'metacarpal');
  const pinkyBase = getFingerJoint(context, 'pinky', 'metacarpal');
  if (!wrist || !indexBase || !pinkyBase) return null;

  const u = new THREE.Vector3().subVectors(indexBase, wrist);
  const v = new THREE.Vector3().subVectors(pinkyBase, wrist);
  if (u.lengthSq() === 0 || v.lengthSq() === 0) return null;

  const normal = new THREE.Vector3().crossVectors(u, v);
  if (normal.lengthSq() === 0) return null;
  if (context.handLabel === 'left') normal.multiplyScalar(-1);
  return normal.normalize();
}

function getPalmRight(context: HandContext) {
  const indexBase = getFingerJoint(context, 'index', 'metacarpal');
  const pinkyBase = getFingerJoint(context, 'pinky', 'metacarpal');
  if (!indexBase || !pinkyBase) return null;
  const right = new THREE.Vector3().subVectors(indexBase, pinkyBase);
  if (context.handLabel === 'left') right.multiplyScalar(-1);
  if (right.lengthSq() === 0) return null;
  return right.normalize();
}

function getPalmUp(context: HandContext) {
  const normal = getPalmNormal(context);
  const right = getPalmRight(context);
  if (!normal || !right) return null;

  const up = new THREE.Vector3().copy(right).cross(normal);
  if (up.lengthSq() === 0) return null;
  return up.normalize();
}

function getAdjacentFingerDistances(context: HandContext) {
  const tips = FINGER_ORDER.map((finger) =>
    getFingerJoint(context, finger, 'tip')
  );
  if (tips.some((tip) => !tip)) {
    return {average: Infinity};
  }
  const distances = [
    tips[0]!.distanceTo(tips[1]!),
    tips[1]!.distanceTo(tips[2]!),
    tips[2]!.distanceTo(tips[3]!),
  ];
  return {average: average(distances)};
}

function getJoint(context: HandContext, jointName: string) {
  return context.joints.get(jointName);
}

function getFingerJoint(
  context: HandContext,
  finger: FingerName,
  suffix: string
) {
  const prefix = FINGER_PREFIX[finger];
  return getJoint(context, `${prefix}-${suffix}`);
}

function fingerAlignmentScore(
  context: HandContext,
  metrics: FingerMetrics,
  palmUp: THREE.Vector3
) {
  const base = metrics.metacarpal ?? getJoint(context, 'wrist');
  if (!base) return 0;
  const direction = new THREE.Vector3().subVectors(metrics.tip, base);
  if (direction.lengthSq() === 0) return 0;
  direction.normalize();
  return clamp01((direction.dot(palmUp) - 0.35) / 0.5);
}

function clamp01(value: number) {
  return THREE.MathUtils.clamp(value, 0, 1);
}

function average(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
