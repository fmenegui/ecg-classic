# Print kit: real photos with exact ground truth

PM-ECG-ID and Kaggle give us photos whose geometry is unknown, so a geometric error and a
reading error look the same. A printed page with fiducial markers removes that: every photo of it
has the exact position of every trace sample.

## Make the kit

    pip install wfdb
    python kit/make_kit.py --out kit_train --pages 40 --folds 1-9 --first-page 0      # for training
    python kit/make_kit.py --out kit_test  --pages 20 --folds 10  --first-page 100    # held out

- Real PTB-XL 1.0.3 ECGs (CC-BY 4.0), 500 Hz. Never a PM-ECG-ID ECG, nor another record of the
  same patient, nor a record whose lead II matches an evaluation ECG (same rule as the training
  bank). One record per patient.
- Layouts 3x4+1R, 3x4, 3x4+3R, 6x2, 6x2+1R, 12x1 in turn; grid red, pink, green, blue, grey or
  orange; calibration pulse left, right or both; black or dark blue trace, 0.25-0.4 mm; labels
  on 85% of the pages, column separators on 40%.
- Four ArUco markers (4x4 dictionary, ids 4p..4p+3 on page p, 14 mm) at the corners: a photo
  tells which page it is and where the page lies. Fold 10 pages start at page 100 so the two kits
  never share marker ids.
- `kit.pdf` to print; `gt/` holds each page's ground truth (record, layout, every printed segment
  in mm, markers, signals).

## Print

- A4, landscape, **100% scale** (turn off "fit to page"). Check the 50 mm bar with a ruler: it
  should measure 50 mm (±0.5).
- Colour printer for the coloured grids; plain office paper is fine.

## Photograph

Every page several times, one condition per photo, the four markers inside the frame whenever
the condition allows (the more markers, the better the ground truth):

1. Flat on a table, phone parallel to the page, daylight.
2. Flat, phone tilted 20-40 degrees (perspective).
3. Flat, rotated 5-35 degrees in the frame.
4. Held in the hand or on a curved surface (bent page).
5. Folded in four and opened again (creases).
6. Crumpled and flattened by hand.
7. Dim room or warm lamp; a shadow across part of the page.
8. From far (page about a quarter of the frame) and very close (part of the page cut off).
9. The page shown on a screen and photographed (moire), if convenient.
10. A scan (flatbed) as the clean reference.

Different phones help. File names do not matter: the markers say which page it is.

## Label the photos

    python kit/read_photo.py --gt kit_train/gt --out labels photos/*.jpg

For each photo: the page, the homography from page mm to photo pixels, the reprojection error and
every segment as a polyline in photo pixels, plus `<photo>_flat.png`, the page brought back flat at
10 px/mm. For flat pages (conditions 1-3, 7-10) this is exact (simulated: 0.05 mm). For bent or
crumpled pages the four markers only give the plane through them; those photos serve to measure
how far the rectification is from flat, and with a grid-based correction to train it.

## What the photos are for

- Measure geometry on real photos: rectification error in mm, horizontal scale per column.
- Measure structure: rows, columns and lead order found against the printed layout.
- Train the geometry and row networks on real photos (the training kit only; the test kit stays
  out of training).
