use opencv::types;
use opencv::{core, features2d, prelude::*};

const MATCH_THRESHOLD: f32 = 0.15;

pub fn knnmatch(
    kps: types::VectorOfKeyPoint,
    next_kps: &types::VectorOfKeyPoint,
    desc: core::Mat,
    next_desc: &core::Mat,
) -> opencv::Result<types::VectorOfVectorOfPoint> {
    // KNN mathcing with BruteForce-Hamming(2)
    let mut matches = types::VectorOfVectorOfDMatch::new();
    let matcher = <dyn features2d::DescriptorMatcher>::create("BruteForce-Hamming(2)");
    matcher?.knn_train_match(&desc, &next_desc, &mut matches, 2, &core::no_array(), false)?;

    // filtering Lowe's ratio test
    let mut pts = types::VectorOfVectorOfPoint::new();
    for m in &matches {
        if m.get(0)?.distance < MATCH_THRESHOLD * m.get(1)?.distance {
            let idx = m.get(0)?.query_idx;
            let next_idx = m.get(0)?.train_idx;
            let kp = kps.get(idx as usize)?.pt();
            let next_kp = next_kps.get(next_idx as usize)?.pt();
            let pt = types::VectorOfPoint::from(vec![
                core::Point::new(kp.x.round() as i32, kp.y.round() as i32),
                core::Point::new(next_kp.x.round() as i32, next_kp.y.round() as i32),
            ]);
            pts.push(pt);
        };
    }
    Ok(pts)
}
