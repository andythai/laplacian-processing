
    Waxholm Space atlas of the Sprague Dawley rat brain, version 4

                  *                *                *

    How to open the atlas in the Mouse BIRN Atlasing Toolkit (MBAT)
---------------------------------------------------------------------------

   1. Download and install MBAT - http://www.loni.usc.edu/Software/MBAT

   2. Start MBAT and open the Viewer Workspace 
  
   3. Select File - Open - 3D atlas and browse to
      MBAT_WHS_SD_rat_atlas_v3/start_atlas/WHS_SD_rat_atlas_v4.atlas

   4. To view the atlas together with the MRI/DTI template, you will need to
      modify the .atlas file. Use a plain text editor e.g. Notepad or TextEdit.
      Instructions are included in the .atlas file as comments.



    Troubleshooting: Not enough memory
---------------------------------------------------------------------------

    MBAT is based on ImageJ and comes with a default setting on maximum memory usage.
    This may not be enough for loading large volumetric images such as the MRI/DTI template.

    1. To prevent MBAT from loading complete images into memory, or if you have limited
     memory available, open the atlas using File - Open - Large atlas in the Viewer
     Workspace. This will only load parts of the atlas to be displayed on screen.

    2. To allow use of more memory:

      a) on Windows: run MBAT with the command line option -J-Xmx1500m (this example allows
           up to 1500 Mb of memory). Keep in mind the memory limit for the 32-bit JVM (1.6Gb).

      b) on Mac OS: edit MBAT 3.1.4.app/Contents/Info.plist, e.g. this allows up to 4.5Gb memory:
            <key>VMOptions</key>
            <string>-Xmx4500m</string>

    Note that it is not advised to allow MBAT to use more than 75% of total system memory.

    See also:
      http://www.ccb.ucla.edu/twiki/bin/view/MouseBIRN/TroubleShooting
      http://imagej.nih.gov/ij/docs/install/osx.html#memory
