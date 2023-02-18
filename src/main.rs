use opencv::core::Size;
use opencv::{core, highgui, imgproc, prelude::*, videoio};

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
                25,
                0.01,
                2.5,
                &core::no_array(),
                1,
                false,
                0.04,
            )?;
            // show
            for f in features {
                println!("{:?}", f);
            }
            highgui::imshow(window, &resized_frame)?;
            // highgui::imshow(window, &gray)?;
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
