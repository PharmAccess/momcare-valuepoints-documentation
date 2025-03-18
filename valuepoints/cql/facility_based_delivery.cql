// /*
// Library [NEW]
// 
// Percentage of pregnant women who delivered at the health facility

// Numerator: Number of pregnant women who delivered at the health facility 
// Denominator: Total number of pregnant women whose records were closed (ANC
  // close form) in the last reporting period due to any of the reasons below:
  //   » live birth
  //   » stillbirth
  //   » miscarriage
  //   » abortion
  //   » woman died
  //   » lost to follow-up
  //   » moved away

// */
// library [NEW]

using FHIR version '4.0.1'

include FHIRHelpers version '4.0.1'
include FHIRCommon called FC
include ANCCommon called AC
include ANCBaseConcepts called BCx
include ANCConcepts called Cx
include ANCBaseDataElements called BaseData
include ANCDataElements called PatientData
include ANCContactDataElements called ContactData
include ANCStratifiers called Stratifiers

code "Facility based delivery": 'ANC.End.14' from "Cx"

parameter "Measurement Period" Interval<Date> default Interval[@2023-01-01, @2025-12-31]

context Patient

// /*
// Initial population: Antenatal clients
// */
define "Initial Population":
  exists (BaseData."Antenatal care case")

// /*
// Numerator: Number of pregnant women who delivered at the health facility 
// */
define Numerator:
  exists(
    [Encounter Cx."Place of delivery Choices"] P
      where P.code ~ "Facility based delivery"	 
  )
 
// /*
// Denominator: Total number of pregnant women whose records were closed (ANC
  // close form) in the last reporting period due to any of the reasons below:
  //   » live birth
  //   » stillbirth
  //   » miscarriage
  //   » abortion
  //   » woman died
  //   » lost to follow-up
  //   » moved away
// */ 
define Denominator:
  exists (
    PatientData."Reason for closing ANC record" R
      where R.value as FHIR.CodeableConcept in Stratifiers."Qualifying Reason for closing ANC record Choices"
  )

