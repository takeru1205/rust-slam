use opencv::types;
use opencv::{core, features2d, highgui, imgproc, prelude::*, videoio};
mod preprocess;
mod read;

const NFEATURES: i32 = 500;
const SCALE_FACTOR: f32 = 1.2;
const NLEVELS: i32 = 8;
const EDGE_THRESHOLD: i32 = 100;
const FIRST_LEVEL: i32 = 1;
const WTA_K: i32 = 3;
const PATCH_SIZE: i32 = 31;
const FAST_THRESHOLD: i32 = 20;
const MATCH_THRESHOLD: f32 = 0.15;

fn run() -> opencv::Result<()> {
    // window
    let window = "SLAM";
    highgui::named_window(window, 1)?;
    // read file
    let file_name = "test.mp4";
    let (mut cam, mut frame) = read::read_file(file_name)?;

    // load frame
    videoio::VideoCapture::read(&mut cam, &mut frame)?;
    // resize and convert to gray scale frame
    let (resized_frame, gray) = preprocess::preprocess(&frame)?;
    println!(
        "cols: {}, rows: {}, channels, {}",
        &resized_frame.cols(),
        &resized_frame.rows(),
        &resized_frame.channels()
    );

    println!(
        "cols: {}, rows: {}, channels, {}",
        &gray.cols(),
        &gray.rows(),
        &gray.channels()
    );

    let detector = <dyn opencv::prelude::ORB>::create(
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

    // extract features
    let mut desc = core::Mat::default();
    // convert point vector to key point vector
    let mut kps = opencv::types::VectorOfKeyPoint::new();
    let mut next_frame = core::Mat::default();

    detector?.detect_and_compute(&gray, &core::no_array(), &mut kps, &mut desc, false)?;

    loop {
        videoio::VideoCapture::read(&mut cam, &mut next_frame)?;
        if next_frame.size()?.width > 0 {
            // resize and convert to gray scale frame
            let (mut next_resized_frame, next_gray) = preprocess::preprocess(&next_frame)?;

            // draw keypoints on gbr image
            let mut next_kps = opencv::types::VectorOfKeyPoint::new();
            let mut next_desc = core::Mat::default();
            let mut next_image_with_keypoints = Mat::default();

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

            next_detector?.detect_and_compute(
                &next_gray,
                &core::no_array(),
                &mut next_kps,
                &mut next_desc,
                false,
            )?;

            features2d::draw_keypoints(
                &next_resized_frame,
                &next_kps,
                &mut next_image_with_keypoints,
                core::Scalar::from([0.0, 255.0, 0.0, 255.0]), // green
                features2d::DrawMatchesFlags::DEFAULT,
            )?;

            // matching
            let mut matches = types::VectorOfVectorOfDMatch::new();
            let matcher = <dyn features2d::DescriptorMatcher>::create("BruteForce-Hamming(2)");
            matcher?.knn_train_match(
                &desc,
                &next_desc,
                &mut matches,
                2,
                &core::no_array(),
                false,
            )?;

            // draw matching lines between preframe and current frame
            // wchich filtered Lowe's ratio test
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

            imgproc::polylines(
                &mut next_resized_frame,
                &pts,
                false,
                core::Scalar::from([0.0, 255.0, 0.0, 255.0]), // green
                2,
                8,
                0,
            )?;

            // image show
            highgui::imshow(window, &next_resized_frame)?;
            // key wait
            let key = highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                videoio::VideoCapture::release(&mut cam)?;
                break;
            }
            kps = next_kps;
            desc = next_desc;
        } else {
            println!("No more frames!");
            videoio::VideoCapture::release(&mut cam)?;
            break ();
        }
    }
    Ok(())
}

fn main() {
    run().unwrap()
}
