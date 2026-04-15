/*
 *
 * to Convert from OpenCV 4xN Mat to 3D glam Vec3
 *
 */

use anyhow::{bail, Result};
use opencv::{core, prelude::*};


fn mat4xn_to_points3d(mat: &core::Mat) -> Result<Vec<[f32; 3]>> {
    let rows = mat.rows();
    let cols = mat.cols();

    if rows != 4 {
        bail!("triangulated_pts must be 4xN, but got {}x{}", rows, cols);
    }

    let typ = mat.typ();

    let mut points = Vec::with_capacity(cols as usize);

    match typ {
        t if t == core::CV_32F => {
            for c in 0..cols {
                let x = *mat.at_2d::<f32>(0, c)?;
                let y = *mat.at_2d::<f32>(1, c)?;
                let z = *mat.at_2d::<f32>(2, c)?;
                let w = *mat.at_2d::<f32>(3, c)?;

                if w.abs() > 1e-6 && x.is_finite() && y.is_finite() && z.is_finite() && w.is_finite()
                {
                    points.push([x / w, y / w, z / w]);
                }
            }
        }
        t if t == core::CV_64F => {
            for c in 0..cols {
                let x = *mat.at_2d::<f64>(0, c)?;
                let y = *mat.at_2d::<f64>(1, c)?;
                let z = *mat.at_2d::<f64>(2, c)?;
                let w = *mat.at_2d::<f64>(3, c)?;

                if w.abs() > 1e-9 && x.is_finite() && y.is_finite() && z.is_finite() && w.is_finite()
                {
                    points.push([(x / w) as f32, (y / w) as f32, (z / w) as f32]);
                }
            }
        }
        _ => {
            bail!("unsupported Mat type for triangulated_pts: {}", typ);
        }
    }

    Ok(points)
}


pub fn mat4xn_to_glam_points(mat: &core::Mat) -> Result<Vec<rerun::external::glam::Vec3>> {
    let raw = mat4xn_to_points3d(mat)?;
    Ok(raw
        .into_iter()
        .map(|p| rerun::external::glam::Vec3::new(p[0], p[1], p[2]))
        .collect())
}
