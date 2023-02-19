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
    // preload frame
    videoio::VideoCapture::read(&mut cam, &mut frame)?;
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

    let detector = <dyn opencv::prelude::ORB>::create(
        500,
        1.2,
        8,
        31,
        0,
        2,
        features2d::ORB_ScoreType::HARRIS_SCORE,
        31,
        20,
    );

    // extract features
    let mut desc = core::Mat::default();
    // convert point vector to key point vector
    let mut kps = opencv::types::VectorOfKeyPoint::new();
    let mut next_frame = core::Mat::default();

    let res = detector?.detect_and_compute(&gray, &core::no_array(), &mut kps, &mut desc, false);
    println!("{:?}", res);

    loop {
        videoio::VideoCapture::read(&mut cam, &mut next_frame)?;
        if next_frame.size()?.width > 0 {
            // resize
            let mut next_resized_frame = Mat::default();
            imgproc::resize(
                &next_frame,
                &mut next_resized_frame,
                Size::default(),
                0.5,
                0.5,
                1,
            )?;
            println!(
                "cols: {}, rows: {}, channels: {}",
                &next_resized_frame.cols(),
                &next_resized_frame.rows(),
                &next_resized_frame.channels()
            );
            // convert to gray scale
            let mut next_gray = Mat::default();
            imgproc::cvt_color(
                &next_resized_frame,
                &mut next_gray,
                imgproc::COLOR_BGR2GRAY,
                0,
            )?;
            println!(
                "cols: {}, rows: {}, channels: {}",
                &next_gray.cols(),
                &next_gray.rows(),
                &next_gray.channels()
            );

            let mut next_kps = opencv::types::VectorOfKeyPoint::new();
            // draw keypoints on gbr image
            let mut next_image_with_keypoints = Mat::default();
            let mut next_desc = core::Mat::default();

            let next_detector = <dyn opencv::prelude::ORB>::create(
                500,
                1.2,
                8,
                31,
                0,
                2,
                features2d::ORB_ScoreType::HARRIS_SCORE,
                31,
                20,
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

            // image show
            highgui::imshow(window, &next_image_with_keypoints)?;
            // key wait
            let key = highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                videoio::VideoCapture::release(&mut cam)?;
                break;
            }
            // kps = next_kps
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
