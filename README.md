# LabelSeq
LabelSeq is an open-Source tool for annotating sequences of real and complex numbers. More precisely, annotations can be added as sequences or intervals indicating features within the original sequence set. Eventually, this tool's purpose is to decompose a Sequence into its contributing components.

## Purpose
LabelSeq is intended to generate ground-truth for machine learning applications. Application domains include:

 - Sensor Time-Series
   - e.g. Labeling physical events like surges in temperature/force/voltage/current/impedance measurements
 - Optical Spectra
   - e.g. Decomposition of Astronomic Observations into emitting and absorption Spectra
 - Economic Data
   - e.g. Labeling Events like pandemic influences in stock prices

## Features

 - High-Performance UI, even with Millions of Datapoints
 - Loading from and storing to well-known file formats
   - Parquet
   - CSV
 - Three-Folded View
   - Items
     - Original Signal
     - Arbitrary Amount of Signal Components
     - Remaining Signal
   - Synchronous Pinching/Zooming
