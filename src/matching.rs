use opencv::{
    core::{self, DMatch, KeyPoint, Point, Vector},
    features2d,
    prelude::*,
};

const MATCH_THRESHOLD: f32 = 0.7;

// calc distance between keypoints
fn calc_distance(points: &Vector<Point>) -> opencv::Result<f32> {
    let p0 = points.get(0)?;
    let p1 = points.get(1)?;
    let dx = (p1.x - p0.x) as f32;
    let dy = (p1.y - p0.y) as f32;
    Ok((dx * dx + dy * dy).sqrt())
}

pub fn knnmatch(
    kps: &Vector<KeyPoint>,
    next_kps: &Vector<KeyPoint>,
    desc: &core::Mat,
    next_desc: &core::Mat,
) -> opencv::Result<(
    // types::VectorOfVectorOfPoint,
    Vector<Vector<Point>>,
    Vector<Point>,
    Vector<Point>,
)> {
    // KNN mathcing with BruteForce-Hamming(2)
    let mut matches = Vector::<Vector<DMatch>>::new();
    let matcher = features2d::DescriptorMatcher::create("BruteForce-Hamming(2)")?;
    matcher.knn_train_match(
        desc,
        next_desc,
        &mut matches,
        2,
        &core::no_array(),
        false,
    )?;

    // filtering Lowe's ratio test
    let mut pts = Vector::<Vector<Point>>::new();
    let mut from_pts = Vector::<Point>::new();
    let mut to_pts = Vector::<Point>::new();

    for m in matches.iter() {
        // knn result can contain fewer than k matches
        if m.len() < 2 {
            continue;
        }

        let m0 = m.get(0)?;
        let m1 = m.get(1)?;

        if m0.distance < MATCH_THRESHOLD * m1.distance {
            let idx = m0.query_idx as usize;
            let next_idx = m0.train_idx as usize;

            let kp = kps.get(idx)?.pt();
            let next_kp = next_kps.get(next_idx)?.pt();

            // To estimate affine
            let from_pt = Point::new(kp.x.round() as i32, kp.y.round() as i32);
            let to_pt = Point::new(next_kp.x.round() as i32, next_kp.y.round() as i32);

            let pt = Vector::<Point>::from_iter([from_pt, to_pt]);

            // if unrealistic distance, do not recognize as keypoint
            if calc_distance(&pt)? >= 30.0 {
                continue;
            }

            pts.push(pt);
            from_pts.push(from_pt);
            to_pts.push(to_pt);
        }
    }

    Ok((pts, from_pts, to_pts))

}
