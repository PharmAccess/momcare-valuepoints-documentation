// /*
// Library ANC.IND.10
// WHO Antenatal Care (ANC) Indicator 10
// Percentage of pregnant women with an ultrasound scan before 24 weeks 

// Numerator: Number of pregnant women who had an ultrasound between 7 weeks and 24 weeks 
// Denominator: Total number of  pregnant women who received an ultrasound

// */

library ANCIND10

using FHIR version '4.0.1'

include FHIRHelpers version '4.0.1' called FHIRHelpers
// include FHIRCommon called FC
// include ANCBaseDataElements called BaseData
// include ANCDataElements called PatientData

// parameter "Measurement Period" Interval<Date> default Interval[@2023-01-01, @2025-12-31]

context Patient

// // /*
// // Initial population: Antenatal clients
// // */
// define "Initial Population":
//   exists (BaseData."Antenatal care case")

// // /*
// // Numerator: Number of pregnant women who had an ultrasound between 7 weeks and 24 weeks 
// // */
// define Numerator:
//   exists (
//     PatientData."Ultrasound scan conducted" P
//       where FC.ToInterval(P.performed) during Interval[start of BaseData."Pregnancy".period + 7 weeks, start of BaseData."Pregnancy".period + 24 weeks]
//     )
 

// // /*
// // Denominator: Total number of  pregnant women who received an ultrasound
// // */
// define Numerator:
//   exists (
//     PatientData."Ultrasound scan conducted" P
// )


