// /*
// Library ANC.IND.1
// WHO Antenatal Care (ANC) Indicator 1
// Percentage of pregnant women with first ANC contact in the first trimester (before 12 weeks of gestation)

// Numerator: Number of pregnant women who had their first ANC contact before 12 weeks (facility level)
// Denominator: Total number of antenatal clients with a first contact

// */
library ANCIND01

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

// parameter "Measurement Period" Interval<Date> default Interval[@2023-01-01, @2025-12-31]

context Patient

// /*
// Initial population: Antenatal clients
// */
define "Initial Population":
  exists (BaseData."Antenatal care case")

// /*
// Numerator: Number of pregnant women with four ANC contacts between gestational age 20 weeks and 43 weeks
// */
define Numerator:
  exists (
    "First antenatal care contact" C
      where ContactData."Gestational age" <= 12 weeks
  )

 

// /*
// Denominator: Number of pregnant women with at least four ANC contacts between gestational age 20 weeks and 43 weeks
// */
define Denominator:
  exists (
    "First antenatal care contact" C
  )


