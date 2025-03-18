// /*
// Library [NEW]
// Percentage of pregnant women with a full ANC profile where the following procedures where completed

--urinalisis
--hiv_screening
--syphilis screening
--blood_pressure
--haemoglobine
--blood_grouping

// Numerator: Number of pregnant women with a full ANC profile during first contact
// Denominator: Total number of antenatal clients with a first contact

// */
// library [new]

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
include WHOCommon called WC

parameter "Measurement Period" Interval<Date> default Interval[@2023-01-01, @2025-12-31]

context Patient

// /*
// Initial population: Antenatal clients
// */
define "Initial Population":
  exists (BaseData."Antenatal care case")

define "First antenatal care contact":
  BaseData."Antenatal care contact" C
    where C.reasonCode in Cx."First Antenatal Care Contact Choices"

// /*
// Numerator: Number of pregnant women with a full ANC profile during first contact
// */
define Numerator:
  exists(
    "First antenatal care contact" C
  )
    and exists(
      PatientData."Urine test ordered" U
        where U.performed starts within 1 day of start "First antenatal care contact".period
    )  
    and exists(
      PatientData."HIV test ordered" HT 
        where HT.performed starts within 1 day of start "First antenatal care contact".period
    )
    and exists(
      PatientData."Syphilis test ordered" ST 
        where ST.performed starts within 1 day of start "First antenatal care contact".period        
    )
    and (
      exists (
        PatienetData."Systolic blood pressure" SP
          where SP.performed starts within 1 day of start "First antenatal care contact".period
      )
      or 
      exists (
        PatientData"Diastolic blood pressure" DP 
          where DP.performed starts within 1 day of start "First antenatal care contact".period
      )
      )
    and exists(
      PatientData."Blood haemoglobin test ordered"  HM 
        where HM.performed starts within 1 day of start "First antenatal care contact".period
    )
    and exists(
      PatienetData."Blood type test date" BTP
        where BTP.performed starts within 1 day of start "First antenatal care contact".period
    )

// /*
// Denominator: Total number of antenatal clients with a first contact
// */
define Denominator:
  exists(
    "First antenatal care contact"
  )


