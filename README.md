
# Extract & Compare Signatures From Two Different Documents (Using SkImage's SSIM)

The implementation involves utilizing the [**SSIM (Structural Similarity) function**](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html) from SkImage To Compare Structural Differences between the Signatures.

<br>

**The source code has 3 separate functions**

1.  `def pdf_jpg(path1, path2):`

2.  `def sigExtract(inputPath, outputPath):`

3.  `def match(path1, path2, countour = False):`
   
<br>

**Function 1 - Convert PDFs to JPGs**<br>
Uses a pre-defined function `convert_from_path` imported from `pdf2image`.  

**Function 2 - Extracts Signatures From Documents**<br>
Uses OpenCV. To be more exact, the `cv2.findContours( )` identifies the contours of the signatures and uses it as a mask to then crop the mask out of its background. 

**Function 3 - Compares Signatures & Establishes A "Percentage Of Similarity"**<br>
Uses the `structural_similarity` function from the `skimage.metrics` to compare the physical differences between the two extracted signatures.