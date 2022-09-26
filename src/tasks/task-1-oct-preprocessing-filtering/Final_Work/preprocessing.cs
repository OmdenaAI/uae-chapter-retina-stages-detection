Bitmap img = (Bitmap)Image.FromFile(file);
                    img = img.Clone(new Rectangle(0, 0, img.Width, img.Height), System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                    img.MakeTransparent();
                  //  pictureBox1.Image = img;
                    //convert image to gray scale
                    Grayscale filter = new Grayscale(0.2125, 0.7154, 0.0721);
                    Bitmap Grayimg = filter.Apply(img);
                  // pictureBox2.Image = Grayimg;
                    //Threshold filter
                    Threshold Binaryfilter = new Threshold(50);
                    Bitmap BinaryImg = Binaryfilter.Apply(Grayimg);
                   // pictureBox3.Image = BinaryImg;
                    //morphology filter opening
                    Opening morphologyFilter = new Opening();
                    Bitmap morphologyImg = morphologyFilter.Apply(BinaryImg);
                   // pictureBox4.Image = morphologyImg;
                    //extract biggest blob from binary image
                    ExtractBiggestBlob BblobFilter = new ExtractBiggestBlob();
                    Bitmap BblobImg = BblobFilter.Apply(morphologyImg);
                  //  pictureBox5.Image = BblobImg;
                    //clone image with shrinked boundaries from left and right side
                    Bitmap CroppedImg = BblobImg.Clone(new Rectangle(20, 0, BblobImg.Width - 40, BblobImg.Height), System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                   // pictureBox6.Image = CroppedImg;