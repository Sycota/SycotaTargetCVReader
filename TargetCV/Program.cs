using System;
using System.IO;
using OpenCvSharp;

var inputPath = args.Length > 0 ? args[0] : Path.Combine(AppContext.BaseDirectory, "input.jpg");
var outputDir = args.Length > 1 ? args[1] : Path.Combine(AppContext.BaseDirectory, "export");
var outputFileName = args.Length > 2 ? args[2] : "output.png";

if (!File.Exists(inputPath))
{
    Console.Error.WriteLine($"Input file not found: {inputPath}");
    Environment.ExitCode = 1;
    return;
}

Mat image = Cv2.ImRead(inputPath, ImreadModes.Color);
if (image.Empty())
{
    Console.Error.WriteLine($"Failed to read image (empty Mat): {inputPath}");
    Environment.ExitCode = 2;
    return;
}

Mat blurredImage = new Mat();

Cv2.MedianBlur(image, blurredImage, 3);

Mat grayImage = new Mat();
Mat thresImage = new Mat();

Cv2.CvtColor(blurredImage, grayImage, ColorConversionCodes.BGR2GRAY);
Cv2.Threshold(grayImage, thresImage, 200.0, 255.0, ThresholdTypes.Binary);

Point[][] contours; 
Cv2.FindContours(thresImage, out contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

Directory.CreateDirectory(outputDir);
Mat croppedTarget; 
if (contours.Length > 0)
{
    var largestContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).First();
    Rect targetBounds = Cv2.BoundingRect(largestContour);

    int margin = 0;
    targetBounds.X = Math.Max(0, targetBounds.X - margin);
    targetBounds.Y = Math.Max(0, targetBounds.Y - margin);
    targetBounds.Width = Math.Min(image.Width - targetBounds.X, targetBounds.Width + (2 * margin));
    targetBounds.Height = Math.Min(image.Height - targetBounds.Y, targetBounds.Height + (2 * margin));

    croppedTarget = image[targetBounds].Clone();

    var croppedPath = Path.Combine(outputDir, "cropped_target.png");
    var croppedOk = Cv2.ImWrite(croppedPath, croppedTarget);

    Console.WriteLine($"Target detected: {targetBounds.Width}x{targetBounds.Height}px at ({targetBounds.X}, {targetBounds.Y})");
    if (croppedOk)
        Console.WriteLine($"Cropped target saved to: {croppedPath}");

    Mat grayImageCropped = new Mat();
    Cv2.CvtColor(croppedTarget, grayImageCropped, ColorConversionCodes.BGR2GRAY);
    Mat blurImage = new Mat();
    Cv2.MedianBlur(grayImageCropped, blurImage, 15);
    Mat threshCropped = new Mat();
    Cv2.Threshold(blurImage, threshCropped, 100, 255, ThresholdTypes.Binary);
    
    Point[][] croppedContours;
    Cv2.FindContours(threshCropped, out croppedContours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

    double targetCenterX = croppedTarget.Width / 2.0;
    double targetCenterY = croppedTarget.Height / 2.0;
    double blackCircleRadiusPixels = 0;

    double minBlackCircleArea = 20000;
    double maxBlackCircleArea = 500000;
    if (croppedContours.Length > 0)
    {
        var blackCircleContour = croppedContours
            .Where(c =>
            {
                double area = Cv2.ContourArea(c);
                return area >= minBlackCircleArea && area <= maxBlackCircleArea;
            })
            .OrderByDescending(c => Cv2.ContourArea(c)).First();
        Moments blackCircleMoments = Cv2.Moments(blackCircleContour);

        if (blackCircleMoments.M00 != 0)
        {
            targetCenterX = blackCircleMoments.M10 / blackCircleMoments.M00;
            targetCenterY = blackCircleMoments.M01 / blackCircleMoments.M00;

            // Calculate radius from area: Area = π*r², so r = sqrt(Area/π)
            double blackCircleArea = Cv2.ContourArea(blackCircleContour);
            blackCircleRadiusPixels = Math.Sqrt(blackCircleArea / Math.PI);

            Console.WriteLine($"Black circle center detected at: ({targetCenterX:F2}, {targetCenterY:F2})");
            Console.WriteLine($"Black circle radius: {blackCircleRadiusPixels:F2}px");
        }
    }


    // ISSF 10m Air Rifle scoring zones (in millimeters from center)
    // Black circle diameter on ISSF target = 170mm, so radius = 85mm
    const double issf10mBlackCircleRadiusMm = 85.0;

    // Calculate pixels per mm conversion
    double pixelsPerMm = blackCircleRadiusPixels > 0 ? blackCircleRadiusPixels / issf10mBlackCircleRadiusMm : 1.0;
    Console.WriteLine($"Conversion: {pixelsPerMm:F4} pixels per mm");

    // ISSF 10m scoring rings (radius in mm from center, cumulative)
    Func<double, double> CalculateDecimalScore = (double distanceMm) =>
    {
        if (distanceMm <= 11.5)
        {
            // 10-ring: subdivide into 0.1 increments
            // 0-1.15mm = 10.9, 1.15-2.3mm = 10.8, etc.
            return 10.9 - (distanceMm / 11.5) * 0.9;
        }
        else if (distanceMm <= 23.0)
            return 9.0 + (1.0 - (distanceMm - 11.5) / 11.5);
        else if (distanceMm <= 34.5)
            return 8.0 + (1.0 - (distanceMm - 23.0) / 11.5);
        else if (distanceMm <= 46.0)
            return 7.0 + (1.0 - (distanceMm - 34.5) / 11.5);
        else if (distanceMm <= 57.5)
            return 6.0 + (1.0 - (distanceMm - 46.0) / 11.5);
        else if (distanceMm <= 69.0)
            return 5.0 + (1.0 - (distanceMm - 57.5) / 11.5);
        else if (distanceMm <= 80.5)
            return 4.0 + (1.0 - (distanceMm - 69.0) / 11.5);
        else if (distanceMm <= 92.0)
            return 3.0 + (1.0 - (distanceMm - 80.5) / 11.5);
        else if (distanceMm <= 103.5)
            return 2.0 + (1.0 - (distanceMm - 92.0) / 11.5);
        else if (distanceMm <= 115.0)
            return 1.0 + (1.0 - (distanceMm - 103.5) / 11.5);
        else
            return 0.0;
    };

    // Filter contours by area to isolate the shot hole (smallest significant contour)
    double minShotArea = 900;
    double maxShotArea = 10000;
    var shotContours = croppedContours
        .Where(c =>
        {
            double area = Cv2.ContourArea(c);
            return area >= minShotArea && area <= maxShotArea;
        })
        .OrderBy(c => Cv2.ContourArea(c))
        .ToArray();
    
    // Create visualization image
    Mat visualizationImage = croppedTarget.Clone();

    if (shotContours.Length > 0)
    {
        // Process each shot contour found
        foreach (var shotContour in shotContours)
        {
            double shotArea = Cv2.ContourArea(shotContour);
            Rect shotBounds = Cv2.BoundingRect(shotContour);

            // Calculate centroid using moments
            Moments m = Cv2.Moments(shotContour);
            double shotCenterX = m.M10 / m.M00;
            double shotCenterY = m.M01 / m.M00;

            // Fit perfect circle around the contour (minimum enclosing circle)
            Point2f circleCenter = new Point2f();
            float circleRadius = 0;
            Cv2.MinEnclosingCircle(shotContour, out circleCenter, out circleRadius);

            // Use the circle's center for scoring (but keep drawing the contour centroid too)
            double distanceFromCenter = Math.Sqrt(
                Math.Pow(circleCenter.X - targetCenterX, 2) +
                Math.Pow(circleCenter.Y - targetCenterY, 2)
            );

            // Convert to millimeters
            double distanceFromCenterMm = distanceFromCenter / pixelsPerMm;

            double decimalScore = CalculateDecimalScore(distanceFromCenterMm);


            Console.WriteLine($"Shot detected: Area={shotArea:F2}px², ContourCentroid=({shotCenterX:F2}, {shotCenterY:F2}), EnclosingCircleCenter=({circleCenter.X:F2}, {circleCenter.Y:F2}), Distance from center={distanceFromCenter:F2}px");
            Console.WriteLine($"Distance from center: {distanceFromCenter:F2}px ({distanceFromCenterMm:F2}mm)");
            Console.WriteLine($"SCORE: {decimalScore:F1}");

            // Draw shot contour in green
            Cv2.DrawContours(visualizationImage, new Point[][] { shotContour }, 0, new Scalar(0, 255, 0), 2);

            // Draw the perfect enclosing circle in blue
            Cv2.Circle(visualizationImage, new Point((int)Math.Round(circleCenter.X), (int)Math.Round(circleCenter.Y)), (int)Math.Round(circleRadius), new Scalar(255, 0, 0), 2);

            // Draw a small marker at the enclosing circle center (cyan)
            Cv2.Circle(visualizationImage, new Point((int)Math.Round(circleCenter.X), (int)Math.Round(circleCenter.Y)), 4, new Scalar(255, 255, 0), -1);

            // Draw circle at contour centroid in red (kept for comparison)
            Cv2.Circle(visualizationImage, new Point((int)shotCenterX, (int)shotCenterY), 5, new Scalar(0, 0, 255), -1);

            // Draw score text on visualization
            Cv2.PutText(visualizationImage, $"Score: {decimalScore:F1}", new Point(20, 40),
                HersheyFonts.HersheyPlain, 2.0, new Scalar(0, 0, 255), 2);
        }

        // Draw target center crosshair in yellow (based on black circle center)
        int centerX = (int)targetCenterX;
        int centerY = (int)targetCenterY;
        int crosshairSize = 20;
        Cv2.Line(visualizationImage, new Point(centerX - crosshairSize, centerY), new Point(centerX + crosshairSize, centerY), new Scalar(0, 255, 255), 2);
        Cv2.Line(visualizationImage, new Point(centerX, centerY - crosshairSize), new Point(centerX, centerY + crosshairSize), new Scalar(0, 255, 255), 2);

        var visualPath = Path.Combine(outputDir, "shot_detection.png");
        Cv2.ImWrite(visualPath, visualizationImage);
        Console.WriteLine($"Visualization saved to: {visualPath}");
    }
    else
    {
        Console.WriteLine("No shot holes detected. Adjust area range or threshold.");
    }
    
    var threshPath = Path.Combine(outputDir, "threshold_cropped.png");
    Cv2.ImWrite(threshPath, threshCropped);
}
else
{
    Console.WriteLine("No target contours detected. Adjust threshold value.");
}

var outputPath = Path.Combine(outputDir, outputFileName);
var ok = Cv2.ImWrite(outputPath, thresImage);

if (!ok)
{
    Console.Error.WriteLine($"Failed to write image: {outputPath}");
    Environment.ExitCode = 3;
    return;
}

Console.WriteLine($"Loaded:  {inputPath}");
Console.WriteLine($"Saved to {outputPath}");