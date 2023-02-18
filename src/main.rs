use opencv::core::Size;
use opencv::{core, features2d, highgui, imgproc, prelude::*, videoio};

fn run() -> opencv::Result<()> {
    // window
    let window = "video capture";
    highgui::named_window(window, 1)?;

    // read file
    let file_name = "test.mp4";
    let mut cam = videoio::VideoCapture::from_file(&file_name, videoio::CAP_ANY)?;
    let opened_file = videoio::VideoCapture::open_file(&mut cam, &file_name, videoio::CAP_ANY)?;
    if !opened_file {
        panic!("Unable to open video file!");
    };
    let mut frame = core::Mat::default();
    let frame_read = videoio::VideoCapture::read(&mut cam, &mut frame)?;
    if !frame_read {
        panic!("Unable to read from video file!");
    };
    let opened = videoio::VideoCapture::is_opened(&mut cam)?;
    println!("Opened? {}", opened);
    if !opened {
        panic!("Unable to open video file!");
    };

    loop {
        videoio::VideoCapture::read(&mut cam, &mut frame)?;
        if frame.size()?.width > 0 {
            // resize
            let mut resized_frame = Mat::default();
            imgproc::resize(&frame, &mut resized_frame, Size::default(), 0.5, 0.5, 1)?;
            println!(
                "cols: {}, rows: {}, channels: {}",
                &resized_frame.cols(),
                &resized_frame.rows(),
                &resized_frame.channels()
            );

            // convert to gray scale
            let mut gray = Mat::default();
            imgproc::cvt_color(&resized_frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
            println!(
                "cols: {}, rows: {}, channels: {}",
                &gray.cols(),
                &gray.rows(),
                &gray.channels()
            );

            // extract features
            let mut features = opencv::types::VectorOfPoint2f::new();
            imgproc::good_features_to_track(
                &gray,
                &mut features,
                500,
                0.01,
                10.0,
                &core::no_array(),
                1,
                false,
                0.04,
            )?;

            // convert point vector to key point vector
            let mut kps = opencv::types::VectorOfKeyPoint::new();
            core::KeyPoint::convert_to(&features, &mut kps, 1.0, 1.0, 0, -1)?;

            for f in features {
                println!("{:?}", f);
            }

            // draw keypoints on gbr image
            let mut image_with_keypoints = Mat::default();
            opencv::opencv_branch_4! {
                let default_draw_matches_flags = features2d::DrawMatchesFlags::DEFAULT;
            }
            opencv::not_opencv_branch_4! {
                let default_draw_matches_flags = features2d::DrawMatchesFlags_DEFAULT;
            }
            features2d::draw_keypoints(
                &resized_frame,
                &kps,
                &mut image_with_keypoints,
                core::Scalar::all(-1f64),
                default_draw_matches_flags,
            )?;

            // image show
            highgui::imshow(window, &image_with_keypoints)?;
            // key wait
            let key = highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                videoio::VideoCapture::release(&mut cam)?;
                break;
            }
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
