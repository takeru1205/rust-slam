use opencv::{core, features2d, prelude::*, types};

const NFEATURES: i32 = 500;
const SCALE_FACTOR: f32 = 1.2;
const NLEVELS: i32 = 8;
const EDGE_THRESHOLD: i32 = 100;
const FIRST_LEVEL: i32 = 1;
const WTA_K: i32 = 3;
const PATCH_SIZE: i32 = 31;
const FAST_THRESHOLD: i32 = 20;

pub fn feature_detect(
    frame: &core::Mat,
    gray: &core::Mat,
) -> opencv::Result<(types::VectorOfKeyPoint, core::Mat)> {
    // draw keypoints on gbr image
    let mut kps = opencv::types::VectorOfKeyPoint::new();
    let mut desc = core::Mat::default();

    if frame.size()?.width > 0 {
        // ORB detector
        let next_detector = <dyn opencv::prelude::ORB>::create(
            NFEATURES,
            SCALE_FACTOR,
            NLEVELS,
            EDGE_THRESHOLD,
            FIRST_LEVEL,
            WTA_K,
            features2d::ORB_ScoreType::HARRIS_SCORE,
            PATCH_SIZE,
            FAST_THRESHOLD,
        );

        next_detector?.detect_and_compute(&gray, &core::no_array(), &mut kps, &mut desc, false)?;
    }
    Ok((kps, desc))
}
