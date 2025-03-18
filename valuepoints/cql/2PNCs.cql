// TO UPDATE TO PNC Visits
// // /*
// // Library ANC.IND.4
// // WHO Antenatal Care (ANC) Indicator 4
// // Percentage of pregnant women with at least four ANC contacts

// // Numerator: Number of pregnant women with four ANC contacts
// // Denominator: Total number of antenatal clients with a first contact

// // */
// library ANCIND04

// using FHIR version '4.0.1'

// include FHIRHelpers version '4.0.1'
// include FHIRCommon called FC
// include ANCCommon called AC
// include ANCBaseConcepts called BCx
// include ANCConcepts called Cx
// include ANCBaseDataElements called BaseData
// include ANCDataElements called PatientData
// include ANCContactDataElements called ContactData
// include ANCStratifiers called Stratifiers

// parameter "Measurement Period" Interval<Date> default Interval[@2023-01-01, @2025-12-31]

// context Patient

// // /*
// // Initial population: Antenatal clients
// // */
// define "Initial Population":
//   exists (BaseData."Antenatal care case")

// // /*
// // Numerator: Number of pregnant women with four ANC contacts between gestational age 20 weeks and 43 weeks
// // */
// define Numerator:
//   exists(
//     BaseData."Antenatal care contact"
//       where Count(BaseData."Antenatal care contact") >= 4
//         and Interval[ContactData."Gestational age" >= 20 weeks, ContactData."Gestational age" <= 43 weeks]    
//   )
 

// // /*
// // Denominator: Number of pregnant women with at least four ANC contacts between gestational age 20 weeks and 43 weeks
// // */
// define Denominator:
//   exists(
//     BaseData."Antenatal care contact"
//       where Interval[ContactData."Gestational age" >= 20 weeks, ContactData."Gestational age" <= 43 weeks]    
//   )


