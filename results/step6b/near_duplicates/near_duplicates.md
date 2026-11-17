# Near-duplicate check: D29 pseudo-label source frames vs. Ski 2DPose evaluation images (D30)

Reference: every frame of the 50 SkiTB training races used in D29 (106,298 frames; 6,307 were kept pseudo-label frames). Queries are hashed, never displayed.

64-bit DCT perceptual hash, Hamming distance. Query image sizes: 1280x720, 1920x1080.

| Control | n | min | median | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| re-encode (query vs. itself at 1280x720, JPEG q75) | 787 | 0 | 0 | 0 | 2 | 2 |
| adjacent SkiTB frames (same clip, 1 frame apart) | 2000 | 0 | 4 | 14 | 20 | 28 |
| null: nearest frame from another race | 2000 | 2 | 12 | 16 | 16 | 18 |

**Threshold T = 14** (max of re-encode p99 and adjacent p95). Null rate at T: 90.55% (limit 5%).

| Partition | images | nearest frame: min | median | images ≤ T | nearest kept frame: min | images ≤ T |
| --- | --- | --- | --- | --- | --- | --- |
| val | 293 | 4 | 14 | 246 | 8 | 128 |
| test | 360 | 6 | 14 | 311 | 8 | 152 |
| followcam | 134 | 4 | 12 | 128 | 8 | 79 |

**Verdict: weak evidence: the hash does not separate footage at T (null rate above 5%).**

The 10 closest (val):
- qxfgw1Kd98A/1/qxfgw1Kd98A_1_00004 → skitb/AL/AL0069/frames/00130.jpg (distance 4, kept pseudo-label frame: False)
- qxfgw1Kd98A/16/qxfgw1Kd98A_16_00005 → skitb/AL/AL0096/frames/00614.jpg (distance 6, kept pseudo-label frame: False)
- qxfgw1Kd98A/16/qxfgw1Kd98A_16_00006 → skitb/AL/AL0096/frames/00614.jpg (distance 6, kept pseudo-label frame: False)
- SVYdH614G_M/5/SVYdH614G_M_5_00002 → skitb/AL/AL0079/frames/01035.jpg (distance 8, kept pseudo-label frame: False)
- qxfgw1Kd98A/14/qxfgw1Kd98A_14_00002 → skitb/AL/AL0093/frames/01580.jpg (distance 8, kept pseudo-label frame: False)
- qxfgw1Kd98A/14/qxfgw1Kd98A_14_00004 → skitb/AL/AL0093/frames/01580.jpg (distance 8, kept pseudo-label frame: False)
- qxfgw1Kd98A/14/qxfgw1Kd98A_14_00010 → skitb/AL/AL0089/frames/01224.jpg (distance 8, kept pseudo-label frame: False)
- qxfgw1Kd98A/19/qxfgw1Kd98A_19_00008 → skitb/AL/AL0050/frames/02301.jpg (distance 8, kept pseudo-label frame: False)
- qxfgw1Kd98A/19/qxfgw1Kd98A_19_00009 → skitb/AL/AL0050/frames/00998.jpg (distance 8, kept pseudo-label frame: False)
- qxfgw1Kd98A/3/qxfgw1Kd98A_3_00003 → skitb/AL/AL0093/frames/01279.jpg (distance 8, kept pseudo-label frame: False)

The 10 closest (test):
- NrWcP1s3QC0/6/NrWcP1s3QC0_6_00001 → skitb/AL/AL0007/frames/01116.jpg (distance 6, kept pseudo-label frame: False)
- uLW74013Wp0/19/uLW74013Wp0_19_00017 → skitb/AL/AL0064/frames/01030.jpg (distance 6, kept pseudo-label frame: False)
- uLW74013Wp0/20/uLW74013Wp0_20_00003 → skitb/AL/AL0043/frames/01389.jpg (distance 6, kept pseudo-label frame: False)
- uLW74013Wp0/15/uLW74013Wp0_15_00006 → skitb/AL/AL0064/frames/01008.jpg (distance 8, kept pseudo-label frame: False)
- uLW74013Wp0/18/uLW74013Wp0_18_00006 → skitb/AL/AL0004/frames/00795.jpg (distance 8, kept pseudo-label frame: False)
- uLW74013Wp0/20/uLW74013Wp0_20_00005 → skitb/AL/AL0056/frames/01447.jpg (distance 8, kept pseudo-label frame: False)
- NrWcP1s3QC0/0/NrWcP1s3QC0_0_00002 → skitb/AL/AL0051/frames/01500.jpg (distance 10, kept pseudo-label frame: False)
- NrWcP1s3QC0/1/NrWcP1s3QC0_1_00000 → skitb/AL/AL0006/frames/00414.jpg (distance 10, kept pseudo-label frame: False)
- NrWcP1s3QC0/1/NrWcP1s3QC0_1_00001 → skitb/AL/AL0070/frames/01510.jpg (distance 10, kept pseudo-label frame: False)
- uLW74013Wp0/0/uLW74013Wp0_0_00006 → skitb/AL/AL0100/frames/01750.jpg (distance 10, kept pseudo-label frame: False)

The 10 closest (followcam):
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00011 → skitb/AL/AL0072/frames/00462.jpg (distance 4, kept pseudo-label frame: False)
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00005 → skitb/AL/AL0100/frames/00442.jpg (distance 8, kept pseudo-label frame: False)
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00029 → skitb/AL/AL0093/frames/00920.jpg (distance 8, kept pseudo-label frame: False)
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00032 → skitb/AL/AL0070/frames/01095.jpg (distance 8, kept pseudo-label frame: False)
- FW1S3VAKui8/0/FW1S3VAKui8_0_00026 → skitb/AL/AL0057/frames/01449.jpg (distance 8, kept pseudo-label frame: False)
- LjXeTB3Wl10/0/LjXeTB3Wl10_0_00001 → skitb/AL/AL0040/frames/01322.jpg (distance 8, kept pseudo-label frame: False)
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00012 → skitb/AL/AL0091/frames/02209.jpg (distance 10, kept pseudo-label frame: False)
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00013 → skitb/AL/AL0051/frames/01322.jpg (distance 10, kept pseudo-label frame: False)
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00018 → skitb/AL/AL0040/frames/01033.jpg (distance 10, kept pseudo-label frame: False)
- 37HtpPpdvp0/0/37HtpPpdvp0_0_00021 → skitb/AL/AL0033/frames/00360.jpg (distance 10, kept pseudo-label frame: False)

