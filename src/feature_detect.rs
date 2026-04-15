use opencv::{
    core::{self, Vector, KeyPoint, Mat},
    features2d::{self, GFTTDetector, ORB},
    prelude::*,
};

const NFEATURES: i32 = 500;
const SCALE_FACTOR: f32 = 1.2;
const NLEVELS: i32 = 8;
const EDGE_THRESHOLD: i32 = 31;
const FIRST_LEVEL: i32 = 0;
const WTA_K: i32 = 2;
const PATCH_SIZE: i32 = 31;
const FAST_THRESHOLD: i32 = 20;

pub fn feature_detect(
    frame: &core::Mat,
    gray: &core::Mat,
) -> opencv::Result<(Vector<KeyPoint>, Mat)> {
    // draw keypoints on gbr image
    let mut kps = Vector::<KeyPoint>::new();
    let mut desc = Mat::default();

    if frame.size()?.width > 0 {
        // feature detection by Good Features to Track
        let mut detector = GFTTDetector::create(1000, 0.01, 1.0, 3, false, 0.04)?;
        detector.detect(gray, &mut kps, &core::no_array())?;

        // ORB description
        let mut descriptor = ORB::create(
            NFEATURES,
            SCALE_FACTOR,
            NLEVELS,
            EDGE_THRESHOLD,
            FIRST_LEVEL,
            WTA_K,
            features2d::ORB_ScoreType::HARRIS_SCORE,
            PATCH_SIZE,
            FAST_THRESHOLD,
        )?;

        descriptor.compute(&gray, &mut kps, &mut desc)?;
    }
    Ok((kps, desc))
}
