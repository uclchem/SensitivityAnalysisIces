 MODULE SurfaceReactions
  USE constants
  USE f2py_constants
  USE network
  IMPLICIT NONE
  REAL(dp) :: surfaceCoverage,totalSwap,bulkLayersReciprocal
  REAL(dp) :: safeMantle,safeBulk
  REAL(dp) :: diffToBindRatio, EDEndothermicityFactor
  LOGICAL :: h2EncounterDesorption, hEncounterDesorption
  LOGICAL :: h2StickingCoeffByh2Coverage, hStickingCoeffByh2Coverage

  !Silicate grain properties for H2 Formation
  REAL(dp),PARAMETER :: SILICATE_MU=0.005D0 ! Fraction of newly formed H2 that stays on the grain surface
  REAL(dp),PARAMETER :: SILICATE_E_S=110.0D0 ! Energy of the saddle point between a physisorbed and a chemisorbed site (K)
  REAL(dp),PARAMETER :: SILICATE_E_H2=320.0D0 ! Desorption energy of H2 molecules (K)
  REAL(dp),PARAMETER :: SILICATE_E_HP=450.0D0 ! Desorption energy of physisorbed H atoms (K)
  REAL(dp),PARAMETER :: SILICATE_E_HC=3.0D4   ! Desorption energy of chemisorbed H atoms (K)
  REAL(dp),PARAMETER :: SILICATE_NU_H2=3.0D12 ! Vibrational frequency of H2 molecules in surface sites (s^-1)
  REAL(dp),PARAMETER :: SILICATE_NU_HC=1.3D13 ! Vibrational frequency of H atoms in their surface sites (s^-1)
  REAL(dp),PARAMETER :: SILICATE_CROSS_SECTION=8.473D-22!*CROSS_SECTION_SCALE ! Silicate grain cross section per H nucleus (cm^-2/nucleus)

  !Graphite grain properties for H2 Formation
  REAL(dp),PARAMETER :: GRAPHITE_MU=0.005D0   ! Fraction of newly formed H2 that stays on the grain surface
  REAL(dp),PARAMETER :: GRAPHITE_E_S=260.0D0  ! Energy of the saddle point between a physisorbed and a chemisorbed site (K)
  REAL(dp),PARAMETER :: GRAPHITE_E_H2=520.0D0 ! Desorption energy of H2 molecules (K)
  REAL(dp),PARAMETER :: GRAPHITE_E_HP=800.0D0 ! Desorption energy of physisorbed H atoms (K)
  REAL(dp),PARAMETER :: GRAPHITE_E_HC=3.0D4   ! Desorption energy of chemisorbed H atoms (K)
  REAL(dp),PARAMETER :: GRAPHITE_NU_H2=3.0D12 ! Vibrational frequency of H2 molecules in surface sites (s^-1)
  REAL(dp),PARAMETER :: GRAPHITE_NU_HC=1.3D13 ! Vibrational frequency of H atoms in their surface sites (s^-1)
  REAL(dp),PARAMETER :: GRAPHITE_CROSS_SECTION=7.908D-22!*CROSS_SECTION_SCALE ! Graphite grain cross section per H nucleus (cm^-2/nucleus)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !Grain surface parameters
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  REAL(dp), PARAMETER :: GAS_DUST_MASS_RATIO=100.0D0,GRAIN_RADIUS=1.0D-5, GRAIN_DENSITY = 3.0D0 ! Mass density of a dust grain
  REAL(dp), PARAMETER :: THERMAL_VEL= SQRT(8.0d0*K_BOLTZ/(PI*AMU)) !Thermal velocity without the factor of SQRT(T/m) where m is moelcular mass in amu

  !reciprocal of fractional abundance of dust grains (we only divide by number density so better to store reciprocal)
  REAL(dp), PARAMETER :: GAS_DUST_DENSITY_RATIO = (4.0D0*PI*(GRAIN_RADIUS**3)*GRAIN_DENSITY*GAS_DUST_MASS_RATIO)/(3.0D0 * AMU)
  !Grain area per h nuclei, values taken from Cazaux & Tielens 2004 via UCL-PDR to match H2 formation rate
  REAL(dp), PARAMETER :: GRAIN_CROSSSECTION_PER_H=0.5D0*(7.908D-22+8.473D-22)
  REAL(dp), PARAMETER :: GRAIN_SURFACEAREA_PER_H=4.0D0*GRAIN_CROSSSECTION_PER_H!2.0*4.0*PI*GRAIN_RADIUS*GRAIN_RADIUS/GAS_DUST_DENSITY_RATIO

  REAL(dp), PARAMETER :: GRAIN_CROSSECTION = PI*GRAIN_RADIUS ** 2

  !Below are values for grain surface reactions
  LOGICAL, PARAMETER :: DIFFUSE_REACT_COMPETITION=.True., GRAINS_HAVE_ICE=.True.
  REAL(dp), PARAMETER :: NUM_MONOLAYERS_IS_SURFACE=2.0D0 ! Number of monolayers to count as surface
  LOGICAL, PARAMETER :: useGarrod2011Transfer=.True. ! Use Garrod 2011 transfer upon net desorption. Otherwise, use Hasegawa 1992
  LOGICAL, PARAMETER :: useCustomReducedMass=.True. ! Use a custom predicted reduced mass instead of the reduced mass of the two
  !! reagents to calculate the tunneling probability.
  REAL(dp), PARAMETER :: CHEMICAL_BARRIER_THICKNESS = 1.40d-8  !gre Parameter used to compute the probability for a surface reaction with 
  !! activation energy to occur through quantum tunneling (Hasegawa et al. Eq 6 (1992).)
  REAL(dp), PARAMETER :: SURFACE_SITE_DENSITY = 1.5d15 ! site density on one grain [cm-2]
  REAL(dp), PARAMETER :: NUM_SITES_PER_GRAIN = GRAIN_RADIUS*GRAIN_RADIUS*SURFACE_SITE_DENSITY*4.0*PI
  
  REAL(dp), PARAMETER :: HH_VDES_PREFACTOR=2.0D0*K_BOLTZ_SI*SURFACE_SITE_DENSITY*1.0D4/(PI*PI*AMU)
  REAL(dp), PARAMETER :: TST_VDES_PREFACTOR = 2.0d0*PI*K_BOLTZ_SI**2*AMU/ &
    (SURFACE_SITE_DENSITY*1.0d4*(HP_SI**3)*1.0d3)


  ! Parameters for Encounter Desorption mechanism (Hincelin et al 2015)
  REAL(dp), PARAMETER :: H2_ON_H2_BINDING_ENERGY=23.0D0 ! Binding energy of H2 on a surface of H2 (Cuppen 2017), used for H2 ED
  REAL(dp), PARAMETER :: H_ON_H2EDEndothermicityFactor_BINDING_ENERGY=45.0D0 ! Binding energy of H on a surface of H2 (Cuppen 2017), used for H ED

  REAL(dp), PARAMETER :: MAX_GRAIN_TEMP=150.0, MIN_SURFACE_ABUND=1.0d-20

  REAL(dp), DIMENSION(nReac) :: desorptionFractionsBare, desorptionFractionsFullCoverage


  REAL(dp) :: HdiffusionBarrier
  REAL(dp), DIMENSION(SIZE(iceList)) :: vdiff, vdes
  LOGICAL :: useCustomDiffusionBarriers
  LOGICAL :: seperateDiffAndDesorbPrefactor, useTSTprefactors, useCustomPrefactors
  LOGICAL :: useMinissaleIceChemdesEfficiency
  
CONTAINS
  !=======================================================================
  !
  !  Calculate the rate of molecular hydrogen (H2) formation on grains
  !  using the treatment of Cazaux & Tielens (2002, ApJ, 575, L29) and
  !  Cazaux & Tielens (2004, ApJ, 604, 222).
  !
  !-----------------------------------------------------------------------
  FUNCTION h2FormEfficiency(gasTemp,dustTemp) RESULT(rate)
    REAL(dp) :: rate
    REAL(dp), INTENT(IN) :: gasTemp,dustTemp

    REAL(dp) :: THERMAL_VELOCITY,STICKING_COEFFICIENT,CROSS_SECTION_SCALE
    REAL(dp) :: FLUX,FACTOR1,FACTOR2,EPSILON
    REAL(dp) :: SILICATE_FORMATION_EFFICIENCY,GRAPHITE_FORMATION_EFFICIENCY
    !  Mean thermal velocity of hydrogen atoms (cm s^-1)
    THERMAL_VELOCITY=1.45D5*SQRT(gasTemp/1.0D2)

    !  Calculate the thermally averaged sticking coefficient of hydrogen atoms on grains,
    !  as given by Hollenbach & McKee (1979, ApJS, 41, 555, eqn 3.7)
    STICKING_COEFFICIENT=1.0D0/(1.0D0+0.04D0*SQRT(gasTemp+dustTemp) &
                    & + 0.2D0*(gasTemp/1.0D2)+0.08D0*(gasTemp/1.0D2)**2)

    FLUX=1.0D-10 ! Flux of H atoms in monolayers per second (mLy s^-1)

    FACTOR1=SILICATE_MU*FLUX/(2*SILICATE_NU_H2*EXP(-SILICATE_E_H2/dustTemp))

   FACTOR2=1.0D0*(1.0D0+SQRT((SILICATE_E_HC-SILICATE_E_S)/(SILICATE_E_HP-SILICATE_E_S)))**2 &
        & /4.0D0*EXP(-SILICATE_E_S/dustTemp)

   EPSILON=1.0D0/(1.0D0+SILICATE_NU_HC/(2*FLUX)*EXP(-1.5D0*SILICATE_E_HC/dustTemp) &
              & *(1.0D0+SQRT((SILICATE_E_HC-SILICATE_E_S)/(SILICATE_E_HP-SILICATE_E_S)))**2)

   SILICATE_FORMATION_EFFICIENCY=1.0D0/(1.0D0+FACTOR1+FACTOR2)*EPSILON


   FACTOR1=GRAPHITE_MU*FLUX/(2*GRAPHITE_NU_H2*EXP(-GRAPHITE_E_H2/dustTemp))

   FACTOR2=1.0D0*(1.0D0+SQRT((GRAPHITE_E_HC-GRAPHITE_E_S)/(GRAPHITE_E_HP-GRAPHITE_E_S)))**2 &
        & /4.0D0*EXP(-GRAPHITE_E_S/dustTemp)

   EPSILON=1.0D0/(1.0D0+GRAPHITE_NU_HC/(2*FLUX)*EXP(-1.5D0*GRAPHITE_E_HC/dustTemp) &
              & *(1.0D0+SQRT((GRAPHITE_E_HC-GRAPHITE_E_S)/(GRAPHITE_E_HP-GRAPHITE_E_S)))**2)

   GRAPHITE_FORMATION_EFFICIENCY=1.0D0/(1.0D0+FACTOR1+FACTOR2)*EPSILON

!  Use the treatment of Cazaux & Tielens (2002, ApJ, 575, L29) and
!  Cazaux & Tielens (2004, ApJ, 604, 222)
   rate=0.5D0*THERMAL_VELOCITY*(SILICATE_CROSS_SECTION*SILICATE_FORMATION_EFFICIENCY &
    & + GRAPHITE_CROSS_SECTION*GRAPHITE_FORMATION_EFFICIENCY)*STICKING_COEFFICIENT

     RETURN
  END FUNCTION h2FormEfficiency

  SUBROUTINE bulkSurfaceExchangeReactions(rate,dustTemperature)
    REAL(dp), INTENT(INOUT) :: rate(*)
    REAL(dp) :: dustTemperature
    IF (THREE_PHASE) THEN
      surfaceCoverage=bulkGainFromMantleBuildUp()
      CALL bulkToSurfaceSwappingRates(rate,bulkswapReacs(1),bulkswapReacs(2),dustTemperature)
      rate(surfSwapReacs(1):surfSwapReacs(2))=surfaceToBulkSwappingRates(dustTemperature)
    END IF
  END SUBROUTINE bulkSurfaceExchangeReactions

  !surface abundance multiplied by this value gives fraction of surface covered by material
  FUNCTION bulkGainFromMantleBuildUp() RESULT(rate)
    REAL(dp) :: rate
    rate=1.0D0/NUM_MONOLAYERS_IS_SURFACE*GAS_DUST_DENSITY_RATIO/NUM_SITES_PER_GRAIN
  END FUNCTION bulkGainFromMantleBuildUp

  FUNCTION surfaceToBulkSwappingRates(dustTemperature) RESULT(rate)
    REAL(dp) ::rate,dustTemperature
    IF ((dustTemperature .gt. MAX_GRAIN_TEMP) .or. (safeMantle .lt. MIN_SURFACE_ABUND)) THEN
        rate = 0.0D0
    ELSE
        rate = 1.0D0
    END IF
  END FUNCTION surfaceToBulkSwappingRates


  SUBROUTINE bulkToSurfaceSwappingRates(rate,idx1,idx2,dustTemperature)
    REAL(dp), INTENT(INOUT) :: rate(*)
    REAL(dp) :: dustTemperature
    INTEGER :: idx1,idx2,i,j
    IF ((dustTemperature .gt. MAX_GRAIN_TEMP) .or. (safeMantle .lt. MIN_SURFACE_ABUND)) THEN
        rate(idx1:idx2) = 0.0D0
    ELSE
        DO i=idx1,idx2
            DO j=lbound(iceList,1),ubound(iceList,1)
                IF (iceList(j) .eq. re1(i)) THEN
                    rate(i)=vdiff(j)*EXP(-bindingEnergy(j)/dustTemperature)
                END IF
            END DO
        END DO
    END IF
  END SUBROUTINE bulkToSurfaceSwappingRates

  !----------------------------------------------------------------------------------------------------
!Reactions on the surface treated by evaluating diffusion rates across grains and accounting
!For competition with chemical desorption. Products remain bound ('DIFF') or are desorbed ('CHEMDES')
!Units of s-1. 
!David Quenard 2017 Arxiv:1711.05184
!----------------------------------------------------------------------------------------------------
FUNCTION getDiffusionBarrier(iceListIndex)
    !! Calculate the diffusion barrier of a species in ice.
    ! For all species, except hydrogen, this is assumed to be a fraction of the binding energy.
    ! For hydrogen, it is this by default, but can be set differently by an option in the input
    INTEGER :: iceListIndex
    REAL(dp) :: getDiffusionBarrier
    IF ((useCustomDiffusionBarriers) .and. &
        (diffusionBarrier(iceListIndex) .ne. 0.0)) THEN
        getDiffusionBarrier = diffusionBarrier(iceListIndex)
    ELSE IF (iceList(iceListIndex) .eq. ngh) THEN
        getDiffusionBarrier = HdiffusionBarrier
    ELSE
        getDiffusionBarrier = diffToBindRatio*bindingEnergy(iceListIndex)
    END IF
END FUNCTION getDiffusionBarrier

FUNCTION reacProb(reacIndx, index1, index2, dustTemperature)
    INTEGER :: reacIndx, index1, index2
    REAL(dp) :: dustTemperature, reducedMass
    REAL(dp) :: reacProb, tunnelProb
    !Calculate classical activation energy barrier exponent
    reacProb = gama(reacIndx)/dustTemperature
    !Calculate quantum activation energy barrier exponent
    reducedMass = reducedMasses(reacIndx)
    IF (reducedMass .eq. 0.0D0) THEN
        WRITE(*,*) "The following reaction had an input reduced mass of 0"
        WRITE(*,*) reacIndx, specname(re1(reacIndx)), specname(re2(reacIndx)), &
            specname(p1(reacIndx)), specname(p2(reacIndx))
        STOP
    END IF
    IF ((.NOT. useCustomReducedMass) .OR. (reducedMass .eq. 0.0D0)) THEN
        ! reducedMasses(reacIndx) should never be 0, but just in case we calculate it here.
        reducedMass = mass(iceList(index1)) * mass(iceList(index2)) / (mass(iceList(index1)) + mass(iceList(index2)))
    END IF
    tunnelProb = 2.0d0 *CHEMICAL_BARRIER_THICKNESS/REDUCED_PLANCK * sqrt(2.0d0*AMU*reducedMass*K_BOLTZ*gama(reacIndx))

    !Choose fastest between classical and tunnelling
    IF (reacProb.GT.tunnelProb) reacProb=tunnelProb
END FUNCTION reacProb

REAL(dp) FUNCTION diffusionReactionRate(reacIndx,dustTemperature)
    REAL(dp) :: reducedMass,tunnelProb,dustTemperature
    REAL(dp) :: diffuseProb,desorbProb,reactionProb,n_dust
    integer :: index1,index2,reacIndx,i


    !want position of species in the grain array but gas phase species aren't in there
    !so store species index
    index1=re1(reacIndx)
    index2=re2(reacIndx)

    !then try to overwrite with position in grain array
    DO i=lbound(iceList,1),ubound(iceList,1)
        IF (iceList(i) .eq. index1) index1 = i
        IF (iceList(i) .eq. index2) index2 = i
    END DO

    !Hasegawa 1992 diffusion rate. Rate that two species diffuse and meet on grain surface
    diffuseProb = vdiff(index1)*exp(-getDiffusionBarrier(index1)/dustTemperature)
    diffuseProb = diffuseProb+ (vdiff(index2)*exp(-getDiffusionBarrier(index2)/dustTemperature))

    !probability a reactant will just desorb
    desorbProb = vdes(index1)*exp(-bindingEnergy(index1)/dustTemperature)
    desorbProb = desorbProb + vdes(index2)*exp(-bindingEnergy(index2)/dustTemperature) 

    !Overall reaction probability is chance of reaction occuring on meeting * diffusion rate
    reactionProb = max(vdiff(index1),vdiff(index2)) * exp(-reacProb(reacIndx, index1, index2, dustTemperature))       


    ! Keff from Garrod & Pauly 2011 and Ruaud+2016
    ! Actual reaction probability is Preac/(Preac+Pevap+Pdiffuse), accounting for the other possible processes
    IF(DIFFUSE_REACT_COMPETITION) THEN
       reactionProb = reactionProb/(reactionProb + desorbProb + diffuseProb)
    ELSE
        reactionProb = 1.0D0
    END IF
    
    !see Eq A1 of Quenard et al. 2018
    !NUM_SITES_PER_GRAIN should be multiplied by n_dust as in A1
    !n_dust=density/GAS_DUST_DENSITY_RATIO so we use the 1/density to cancel the density in odes.f90 and drop it here
    diffusionReactionRate=alpha(reacIndx) *reactionProb* diffuseProb*GAS_DUST_DENSITY_RATIO/NUM_SITES_PER_GRAIN
END FUNCTION diffusionReactionRate


SUBROUTINE updateVdiffAndVdes(gasTemp, dustTemp, vdiff, vdes)
    REAL(dp), INTENT(IN) :: gasTemp, dustTemp

    REAL(dp), INTENT(OUT) :: vdiff(SIZE(iceList)), vdes(SIZE(iceList))

    ! inertiaProducts are scaled up by 1e50 in Makerates, because otherwise they are too small to deal with numerically.
    ! We need to scale them down again here. 
    ! Care needs to be taken to keep this factor the same as in uclchem.makerates.species
    REAL(dp), PARAMETER :: scaleFactor = 1d-50

    REAL(dp) :: estimatedInertiaProduct
    integer :: i, j

    IF (.NOT. useTSTprefactors) THEN
        DO i=LBOUND(iceList,1),UBOUND(iceList,1)
            j = iceList(i)
            ! Original treatment by Hasegawa et al, 1992.
            vdes(i) = SQRT(HH_VDES_PREFACTOR*bindingEnergy(i)*mass(j))
            vdiff(i) = vdes(i)
        END DO
    ELSE
        ! We assume that the desorbed species has not thermalized, and so we use the dust temperature
        ! for both kB*T/h and rotational partition functions.
        vdes(:) = TST_VDES_PREFACTOR * dustTemp * dustTemp
        
        DO  i=lbound(iceList,1),ubound(iceList,1)
            j=iceList(i)
            IF (atomCounts(j) .eq. 1) THEN ! Atomic species, has no rotational partition function
                ! vdes = kB*T/h * q_tr,2D
                vdes(i) = vdes(i) * mass(j)
            ELSE IF (inertiaProducts(i) .gt. 0.0) THEN ! Custom supplied 1/sigma*SQRT(Ix*Iy*Iz)
                IF (moleculeIsLinear(i)) THEN ! Linear molecule, e.g. H2, OH or CO2
                    ! vdes = kB*T/h * q_tr,2D * q_rot,3D(lin)
                    vdes(i) = vdes(i) * mass(j)*scaleFactor * &
                        SQRT(PI) / (HP_SI**2)*(8.0D0*PI**2*K_BOLTZ_SI*dustTemp)*&
                        inertiaProducts(i)
                ELSE
                    ! vdes = kB*T/h * q_tr,2D * q_rot,3D(nonlin)
                    vdes(i) = vdes(i) * mass(j) *scaleFactor* &
                        SQRT(PI) / (HP_SI**3)*(8.0D0*PI**2*K_BOLTZ_SI*dustTemp)**(3.0D0/2.0D0)*&
                        inertiaProducts(i)
                END IF
            ELSE
                IF (atomCounts(j) .eq. 2) THEN
                    ! Need some way to calculate reduced mass.
                    ! Can also be done beforehand, for all diatomic species?
                    ! Then just assume some distance between the two atoms, like 1.5 Angstrom and calculate I
                    ! We can get symmetry factor from whether there is a "2" in the species name.
                        ! If there is, symmetry factor = 2
                        ! If there isn't, symmetry factor = 1
                ELSE
                    ! Since we don't have any information about the structure of the molecule,
                    ! we will assume that the molecule is nonlinear
                    ! Fitted function to estimate inertia product.
                    estimatedInertiaProduct = 2.35425621D-21*EXP(1.04448580D-01*mass(j))
                    ! vdes = kB*T/h * q_tr,2D * q_rot,3D(nonlin)
                    vdes(i) = vdes(i) * mass(j) *scaleFactor* &
                        SQRT(PI) / (HP_SI**3)*(8.0D0*PI**2*K_BOLTZ_SI*dustTemp)** &
                        (3.0D0/2.0D0)*estimatedInertiaProduct
                END IF
            END IF
        END DO

        IF (seperateDiffAndDesorbPrefactor) THEN
            ! Under stationary adsorbate assumption, q^TS = q^RS, so vdiff = kB*T/h
            vdiff = K_BOLTZ_SI * dustTemp / HP_SI
        ELSE
            vdiff = vdes
        END IF
    END IF

    IF (useCustomPrefactors) THEN
        DO i=LBOUND(iceList, 1), UBOUND(iceList, 1)
            IF (customVdiff(i) .gt. 0.0) vdiff(i) = customVdiff(i)
            IF (customVdes(i) .gt. 0.0) vdes(i) = customVdes(i)
        END DO
    END IF

END SUBROUTINE updateVdiffAndVdes

! ---------------------------------------------------------------------
!  Chemical Reactive Desorption (CRD)
! David Quenard 2017 Arxiv:1711.05184
! From Minissalle+ 2016 and Vasyunin+ 2016
! ---------------------------------------------------------------------
REAL(dp) FUNCTION getDesorptionFractionBare(reacIndx, LHDESindex) RESULT(desorptionFractionBare)
    integer :: reacIndx,reactIndex1,reactIndex2,degreesOfFreedom,i,j
    integer :: productIndex(4)

    REAL(dp) :: deltaEnthalpy,maxBindingEnergy,epsilonCd,productEnthalpy
    REAL(dp), PARAMETER :: EFFECTIVE_SURFACE_MASS = 120.0
    REAL(dp) :: bindingEnergyDesorbingSpec

    integer :: desorbingIndex, desorbingOnGrainIndex, LHDESindex, desorbingIceListIndex

     IF (.NOT.(ANY(iceList .eq. re1(reacIndx)) .OR. (ANY(iceList .eq. re2(reacIndx))))) THEN
        ! Gasphase reactions do not need to be calculated, should be 0
        desorptionFractionBare = 0.0d0
        RETURN
     END IF
     IF (ANY(bulkList .eq. re1(reacIndx))) THEN
         ! No chemical desorption from bulk ice allowed
         desorptionFractionBare = 0.0d0
         RETURN
     END IF
    
    ! WRITE(*,*) "    Calculating bare desorption fraction"


    !Get indices of grain surface version of products products 
    productIndex=0
    !Arrays like binding energy and formation enthalpy are indexed by position in iceList
    !rather than species list. Need to find position in grain list where every reactant and product appears
    !bearing in mind that Eley-Rideal reactions can have reactants in gas phase and CHEMDES has products in gas
    DO i=lbound(iceList,1),ubound(iceList,1)
        !check grain lists for reactants
        IF (iceList(i) .eq. re1(reacIndx)) reactIndex1 = i
        IF (gasiceList(i) .eq. re1(reacIndx)) reactIndex1 = i
        !check equivalent gas list in case of ER reaction.
        IF (iceList(i) .eq. re2(reacIndx)) reactIndex2 = i
        IF (gasiceList(i) .eq. re2(reacIndx)) reactIndex2 = i

        IF (iceList(i) .eq. p1(reacIndx)) productIndex(1) = i
        IF (iceList(i) .eq. p2(reacIndx)) productIndex(2) = i
        IF (iceList(i) .eq. p3(reacIndx)) productIndex(3) = i
        IF (iceList(i) .eq. p4(reacIndx)) productIndex(4) = i

        ! WRITE(*,*) "Gas ice list", gasiceList(i), p1(reacIndx)
        ! WRITE(*,*) "Ice list", iceList(i), p1(reacIndx)

        IF (gasiceList(i) .eq. p1(reacIndx)) productIndex(1) = i
        IF (gasiceList(i) .eq. p2(reacIndx)) productIndex(2) = i
        IF (gasiceList(i) .eq. p3(reacIndx)) productIndex(3) = i
        IF (gasiceList(i) .eq. p4(reacIndx)) productIndex(4) = i

    END DO

    ! WRITE(*,*) "    PRODUCT INDECES:", productIndex

    IF (p2(reacIndx) .eq. 9999) THEN
        ! Only one product, and so that one product is desorbing
        desorbingIndex = 1
        desorbingOnGrainIndex = p1(LHDEScorrespondingLHreacs(LHDESindex))
    ELSE IF (p1(LHDEScorrespondingLHreacs(LHDESindex)) .ne. p1(reacIndx)) THEN ! p1 is desorbing
        desorbingIndex = 1
        desorbingOnGrainIndex = p1(LHDEScorrespondingLHreacs(LHDESindex))
    ELSE IF (p2(LHDEScorrespondingLHreacs(LHDESindex)) .ne. p2(reacIndx)) THEN ! p2 is desorbing
        desorbingIndex = 2
        desorbingOnGrainIndex = p2(LHDEScorrespondingLHreacs(LHDESindex))
    ELSE IF (p3(LHDEScorrespondingLHreacs(LHDESindex)) .ne. p3(reacIndx)) THEN ! p3 is desorbing
        desorbingIndex = 3
        desorbingOnGrainIndex = p3(LHDEScorrespondingLHreacs(LHDESindex))
    ELSE
        WRITE(*,*) "COULD NOT DETERMINE DESORBING PRODUCT INDEX OF REACTION:"
        WRITE(*,*) specName(re1(reacIndx)), specName(re2(reacIndx)), "->", &
            specName(p1(reacIndx)), specName(p2(reacIndx)), specName(p3(reacIndx))
        WRITE(*,*) "LHDES INDEX:", LHDESindex
        WRITE(*,*) "REAC INDEX:", reacIndx
        WRITE(*,*) "CORRESPONDING LH INDEX:", LHDEScorrespondingLHreacs(LHDESindex)
        WRITE(*,*) "CORRESPONDING LH REACTION:"
        WRITE(*,*) specName(re1(LHDEScorrespondingLHreacs(LHDESindex))), &
            specName(re2(LHDEScorrespondingLHreacs(LHDESindex))), "->", &
            specName(p1(LHDEScorrespondingLHreacs(LHDESindex))), &
            specName(p2(LHDEScorrespondingLHreacs(LHDESindex))), &
            specName(p3(LHDEScorrespondingLHreacs(LHDESindex)))
        STOP
    END IF



    ! Now we know which product desorbs, we just have to calculate bare desorption prob using Minissale et al 2016.

    ! WRITE(*,*) "    Calculating reaction enthalpy"

    desorbingIceListIndex = 0
    productEnthalpy = 0.0D0
    DO i = 1,4
        IF (productIndex(i) .ne. 0) THEN
            IF (i .eq. desorbingIndex) THEN
                DO j = LBOUND(iceList, 1), UBOUND(iceList, 1)
                    IF (iceList(j) .eq. desorbingOnGrainIndex) THEN
                        desorbingIceListIndex = j
                        productEnthalpy = productEnthalpy + formationEnthalpy(j)
                    END IF
                END DO
            ELSE
                productEnthalpy = productEnthalpy + formationEnthalpy(productIndex(i))
            END IF
        END IF
    END DO

    ! WRITE(*,*) "    DESORBING PRODUCT INDEX:", desorbingIndex
    ! WRITE(*,*) "    DESORBING SPECIES INDEX:", desorbingOnGrainIndex
    ! WRITE(*,*) "    DESORBING SPECIES NAME:", specname(desorbingOnGrainIndex)

    IF (desorbingIceListIndex .eq. 0) THEN
        WRITE(*,*) "COULD NOT DETERMINE ICE LIST INDEX OF PRODUCT ON GRAIN OF REACTION:"
        WRITE(*,*) specName(re1(reacIndx)), specName(re2(reacIndx)), "->", &
            specName(p1(reacIndx)), specName(p2(reacIndx)), specName(p3(reacIndx))
        WRITE(*,*) "DESORBING PRODUCT INDEX:", desorbingIndex
        WRITE(*,*) "DESORBING SPECIES INDEX:", desorbingOnGrainIndex
        WRITE(*,*) "DESORBING SPECIES NAME:", specname(desorbingOnGrainIndex)
        WRITE(*,*) "LHDES INDEX:", LHDESindex
        WRITE(*,*) "REAC INDEX:", reacIndx
        WRITE(*,*) "CORRESPONDING LH INDEX:", LHDEScorrespondingLHreacs(LHDESindex)
        WRITE(*,*) "CORRESPONDING LH REACTION:"
        WRITE(*,*) specName(re1(LHDEScorrespondingLHreacs(LHDESindex))), &
            specName(re2(LHDEScorrespondingLHreacs(LHDESindex))), "->", &
            specName(p1(LHDEScorrespondingLHreacs(LHDESindex))), &
            specName(p2(LHDEScorrespondingLHreacs(LHDESindex))), &
            specName(p3(LHDEScorrespondingLHreacs(LHDESindex)))
        STOP
    END IF

    deltaEnthalpy = productEnthalpy - (formationEnthalpy(reactIndex1) + formationEnthalpy(reactIndex2))
    ! If deltaEnthalpy > 0: endothermic
    ! If deltaEnthalpy < 0: exothermic, energy released to environment

    if (deltaEnthalpy .gt. 0.0) THEN
        ! Endothermic reactions do not induce chemical desorption
        desorptionFractionBare = 0.0d0
        RETURN
    END IF

    ! Now we use deltaEnthalpy as a measure of exothermicity, i.e. the amount of energy released
    deltaEnthalpy = -deltaEnthalpy

    !Convert from kcal to J, from J to K and from moles-1 to reactions-1
    deltaEnthalpy = deltaEnthalpy*KCAL_TO_JOULE/(K_BOLTZ_SI*N_AVOGADRO)


    ! WRITE(*,*) "    Desorbing product index:", desorbingOnGrainIndex
    ! WRITE(*,*) "    Desorbing product mass:", mass(desorbingOnGrainIndex)
    ! WRITE(*,*) "    Desorbing product ice list index:", desorbingIceListIndex
    bindingEnergyDesorbingSpec = bindingEnergy(desorbingIceListIndex)
    ! WRITE(*,*) "    Desorbing product binding energy:", bindingEnergyDesorbingSpec
    IF (deltaEnthalpy .lt. bindingEnergyDesorbingSpec) THEN
        ! WRITE(*,*) "    Returning value"
        desorptionFractionBare = 0.0d0
        RETURN
    END IF


    epsilonCd = mass(desorbingOnGrainIndex)
    
    !epsilonCd is the fraction of kinetic energy kept my the product when it collides with grain surface
    epsilonCd = ((epsilonCd - EFFECTIVE_SURFACE_MASS) / (epsilonCd + EFFECTIVE_SURFACE_MASS))**2

    degreesOfFreedom = 3 * atomCounts(desorbingOnGrainIndex)

    desorptionFractionBare = exp((-bindingEnergyDesorbingSpec*REAL(degreesOfFreedom)) / (epsilonCd * deltaEnthalpy))
    ! WRITE(*,*) "    Returning value"

    ! IF ((trim(specName(re1(reacIndx))) .eq. "#SO2") .and. (trim(specName(re2(reacIndx))) .eq. "#C") & 
    !     .and. (trim(specName(p1(reacIndx))) .eq. "#SO")) THEN
    !     WRITE(*,*) specName(re1(reacIndx)), specName(re2(reacIndx)), "->", &
    !         specName(p1(reacIndx)), specName(p2(reacIndx)), specName(p3(reacIndx))
    !     WRITE(*,*) "LHDES INDEX:", LHDESindex
    !     WRITE(*,*) "REAC INDEX:", reacIndx
    !     WRITE(*,*) "CORRESPONDING LH INDEX:", LHDEScorrespondingLHreacs(LHDESindex)
    !     WRITE(*,*) "CORRESPONDING LH REACTION:"
    !     WRITE(*,*) specName(re1(LHDEScorrespondingLHreacs(LHDESindex))), &
    !         specName(re2(LHDEScorrespondingLHreacs(LHDESindex))), "->", &
    !         specName(p1(LHDEScorrespondingLHreacs(LHDESindex))), &
    !         specName(p2(LHDEScorrespondingLHreacs(LHDESindex))), &
    !         specName(p3(LHDEScorrespondingLHreacs(LHDESindex)))
    !     WRITE(*,*) "PRODUCT ENTHALPY", productEnthalpy
    !     WRITE(*,*) "DELTA ENTHALPY", deltaEnthalpy
    !     WRITE(*,*) "DESORBING MASS", mass(desorbingOnGrainIndex)
    !     WRITE(*,*) "EPSILON CD", epsilonCd
    !     WRITE(*,*) "DESGREES OF FREEDOM", degreesOfFreedom
    !     WRITE(*,*) "BINDING ENERGY OF DESORBING PRODUCT", bindingEnergyDesorbingSpec
    !     WRITE(*,*) "DESORPTION FRACTION BARE", desorptionFractionBare
    !     STOP
    ! END IF
END FUNCTION getDesorptionFractionBare

FUNCTION getDesorptionFractionFullCoverage(reacIndx, LHDESindex) RESULT (desorptionFractionFullCoverage)
    integer :: reacIndx,reactIndex1,reactIndex2,degreesOfFreedom,i,j
    integer :: productIndex(4)

    REAL(dp) :: deltaEnthalpy,maxBindingEnergy,epsilonCd,productEnthalpy
    REAL(dp), PARAMETER :: EFFECTIVE_SURFACE_MASS = 120.0
    REAL(dp) :: bindingEnergyDesorbingSpec, chi
    integer :: desorbingIndex, desorbingOnGrainIndex, LHDESindex, desorbingIceListIndex
    LOGICAL :: twoProductReaction

    REAL(dp) :: desorptionFractionFullCoverage

    IF (.NOT.(ANY(iceList .eq. re1(reacIndx)) .OR. (ANY(iceList .eq. re2(reacIndx))))) THEN
       ! Gasphase reactions do not need to be calculated, should be 0
       desorptionFractionFullCoverage = 0.0d0
       RETURN
    END IF
    IF (ANY(bulkList .eq. re1(reacIndx))) THEN
        ! No chemical desorption from bulk ice allowed
        desorptionFractionFullCoverage = 0.0d0
        RETURN
    END IF
    
    IF (useMinissaleIceChemdesEfficiency) THEN
        desorptionFractionFullCoverage = desorptionFractionsBare(reacIndx)/10.0D0    !< See Minisalle et al. 2016 for icy grain surface.
        ! Special case of OH+H, O+H, N+N on ices, see same paper
        if (re1(reacIndx).eq.ngn.and.re2(reacIndx).eq.ngn) desorptionFractionFullCoverage = 0.5D0
        if ((re1(reacIndx).eq.ngo.and.re2(reacIndx).eq.nh) &
            &.or. (re1(reacIndx).eq. nh.and.re2(reacIndx).eq.ngo)) desorptionFractionFullCoverage = 0.3D0
        if ((re1(reacIndx).eq.ngoh.and.re2(reacIndx).eq.nh) &
            &.or. (re1(reacIndx).eq.nh.and.re2(reacIndx).eq.ngoh)) desorptionFractionFullCoverage = 0.25D0
        RETURN
    ENDIF

    ! Just for testing purposes.
    desorptionFractionFullCoverage = desorptionFractionsBare(reacIndx)/10.0D0    !< See Minisalle et al. 2016 for icy grain surface.
    ! Special case of OH+H, O+H, N+N on ices, see same paper
    if (re1(reacIndx).eq.ngn.and.re2(reacIndx).eq.ngn) desorptionFractionFullCoverage = 0.5D0
    if ((re1(reacIndx).eq.ngo.and.re2(reacIndx).eq.nh) &
        &.or. (re1(reacIndx).eq. nh.and.re2(reacIndx).eq.ngo)) desorptionFractionFullCoverage = 0.3D0
    if ((re1(reacIndx).eq.ngoh.and.re2(reacIndx).eq.nh) &
        &.or. (re1(reacIndx).eq.nh.and.re2(reacIndx).eq.ngoh)) desorptionFractionFullCoverage = 0.25D0

    !Get indices of grain surface version of products products 
    productIndex=0
    !Arrays like binding energy and formation enthalpy are indexed by position in iceList
    !rather than species list. Need to find position in grain list where every reactant and product appears
    !bearing in mind that Eley-Rideal reactions can have reactants in gas phase and CHEMDES has products in gas
    DO i=lbound(iceList,1),ubound(iceList,1)
        !check grain lists for reactants
        IF (iceList(i) .eq. re1(reacIndx)) reactIndex1 = i
        IF (gasiceList(i) .eq. re1(reacIndx)) reactIndex1 = i
        !check equivalent gas list in case of ER reaction.
        IF (iceList(i) .eq. re2(reacIndx)) reactIndex2 = i
        IF (gasiceList(i) .eq. re2(reacIndx)) reactIndex2 = i

        IF (iceList(i) .eq. p1(reacIndx)) productIndex(1) = i
        IF (iceList(i) .eq. p2(reacIndx)) productIndex(2) = i
        IF (iceList(i) .eq. p3(reacIndx)) productIndex(3) = i
        IF (iceList(i) .eq. p4(reacIndx)) productIndex(4) = i

        IF (gasiceList(i) .eq. p1(reacIndx)) productIndex(1) = i
        IF (gasiceList(i) .eq. p2(reacIndx)) productIndex(2) = i
        IF (gasiceList(i) .eq. p3(reacIndx)) productIndex(3) = i
        IF (gasiceList(i) .eq. p4(reacIndx)) productIndex(4) = i
    END DO

    IF (p2(reacIndx) .eq. 9999) THEN
        ! Only one product, and so that one product is desorbing
        desorbingIndex = 1
        desorbingOnGrainIndex = p1(LHDEScorrespondingLHreacs(LHDESindex))
        twoProductReaction = .False.
    ELSE IF (p1(LHDEScorrespondingLHreacs(LHDESindex)) .ne. p1(reacIndx)) THEN ! p1 is desorbing
        desorbingIndex = 1
        desorbingOnGrainIndex = p1(LHDEScorrespondingLHreacs(LHDESindex))
        twoProductReaction = .True.
    ELSE IF (p2(LHDEScorrespondingLHreacs(LHDESindex)) .ne. p2(reacIndx)) THEN ! p2 is desorbing
        desorbingIndex = 2
        desorbingOnGrainIndex = p2(LHDEScorrespondingLHreacs(LHDESindex))
        twoProductReaction = .True.
    ELSE IF (p3(LHDEScorrespondingLHreacs(LHDESindex)) .ne. p3(reacIndx)) THEN ! p3 is desorbing
        desorbingIndex = 3
        desorbingOnGrainIndex = p3(LHDEScorrespondingLHreacs(LHDESindex))
        twoProductReaction = .True.
    ELSE
        WRITE(*,*) "COULD NOT DETERMINE DESORBING PRODUCT INDEX OF REACTION:"
        WRITE(*,*) specName(re1(reacIndx)), specName(re2(reacIndx)), "->", &
            specName(p1(reacIndx)), specName(p2(reacIndx)), specName(p3(reacIndx))
        STOP
    END IF


    ! Now we know which product desorbs, we just have to calculate bare desorption prob using Minissale et al 2016.

    productEnthalpy = 0.0D0
    DO i = 1,4
        IF (productIndex(i) .ne. 0) THEN
            IF (i .eq. desorbingIndex) THEN
                DO j = LBOUND(iceList, 1), UBOUND(iceList, 1)
                    IF (iceList(j) .eq. desorbingOnGrainIndex) THEN
                        desorbingIceListIndex = j
                        productEnthalpy = productEnthalpy + formationEnthalpy(j)
                    END IF
                END DO
            ELSE
                productEnthalpy = productEnthalpy + formationEnthalpy(productIndex(i))
            END IF
        END IF
    END DO

    deltaEnthalpy = productEnthalpy - (formationEnthalpy(reactIndex1) + formationEnthalpy(reactIndex2))
    ! If deltaEnthalpy > 0: endothermic
    ! If deltaEnthalpy < 0: exothermic, energy released to environment

    if (deltaEnthalpy .gt. 0.0) THEN
        ! Endothermic reactions do not induce chemical desorption
        desorptionFractionFullCoverage = 0.0d0
        RETURN
    END IF

    ! Now we use deltaEnthalpy as a measure of exothermicity, i.e. the amount of energy released
    deltaEnthalpy = -deltaEnthalpy

    !Convert from kcal to J, from J to K and from moles-1 to reactions-1
    deltaEnthalpy = deltaEnthalpy*KCAL_TO_JOULE/(K_BOLTZ_SI*N_AVOGADRO)

    bindingEnergyDesorbingSpec = bindingEnergy(desorbingIceListIndex)
    IF (deltaEnthalpy .lt. bindingEnergyDesorbingSpec) THEN
        desorptionFractionFullCoverage = 0.0d0
        RETURN
    END IF

    IF (.NOT. twoProductReaction) THEN
        chi = 0.07D0 ! chi_1 approx 0.07, Furuya et al, 2022
    ELSE
        IF (desorbingIndex .eq. 1) THEN
            chi = 0.2D0 * mass(p2(reacIndx)) / (mass(p1(reacIndx))+mass(p2(reacIndx))) ! Assume that chi_2 = 0.2
        ELSE IF (desorbingIndex .eq. 2) THEN
            chi = 0.2D0 * mass(p1(reacIndx)) / (mass(p1(reacIndx))+mass(p2(reacIndx)))
        ELSE
            WRITE(*,*) "FREDON 2021 METHOD FOR CHEMICAL DESORPTION IS NOT VALID FOR DESORBINDEX > 2"
            STOP
        END IF
    END IF
    IF (chi * deltaEnthalpy - bindingEnergyDesorbingSpec .lt. 0.0) THEN
        desorptionFractionFullCoverage = 0.0D0
    ELSE
        desorptionFractionFullCoverage = 0.5D0*(1.0D0-EXP(-(chi * deltaEnthalpy - &
            bindingEnergyDesorbingSpec)/(3.0D0*bindingEnergyDesorbingSpec)))
    END IF
END FUNCTION getDesorptionFractionFullCoverage

REAL(dp) FUNCTION desorptionFractionIncludingIce(reacIndx, numMonolayers)
    INTEGER :: reacIndx
    REAL(dp) :: numMonolayers

    REAL(dp) :: desorptionFractionBare
    REAL(dp) :: desorptionFractionFullCoverage
    
    desorptionFractionBare = desorptionFractionsBare(reacIndx)
    IF (.NOT. GRAINS_HAVE_ICE) THEN
        ! If we do not simulate with ice, return bare grain desorption efficiency
        desorptionFractionIncludingIce = desorptionFractionBare
        RETURN
    END IF
    

    desorptionFractionFullCoverage = desorptionFractionsFullCoverage(reacIndx)

 
    desorptionFractionIncludingIce = desorptionFractionBare + &
        (desorptionFractionFullCoverage-desorptionFractionBare)*MIN(1.0D0, numMonolayers)
END FUNCTION desorptionFractionIncludingIce

REAL(dp) FUNCTION EncounterDesorptionRate(reacIndx,dustTemperature)
    REAL(dp) :: dustTemperature
    REAL(dp) :: meetProb,desorbProb,diffuseProb
    
    integer :: index1,index2,reacIndx,i

    !want position of species in the grain array but gas phase species aren't in there
    !so store species index
    index1=re1(reacIndx)

    !then try to overwrite with position in grain array
    DO i=lbound(iceList,1),ubound(iceList,1)
        IF (iceList(i) .eq. index1) index1 = i
        IF (iceList(i) .eq. ngh2) index2 = i
    END DO
    
    !Hasegawa 1992 diffusion rate. Rate that two species diffuse and meet on grain surface
    meetProb = vdiff(index1)*exp(-getDiffusionBarrier(index1)/dustTemperature)
    meetProb = meetProb + (vdiff(index2)*exp(-getDiffusionBarrier(index2)/dustTemperature))

    !Adjust the probability of H2 moving onto H2 by the energy required to move of of H2O, onto H2.
    !This is ASSUMED to be the difference in binding energies times a factor between 0 and 1.
    IF (EDEndothermicityFactor .ne. 0.0) THEN
        meetProb = meetProb &
        & * exp(-EDEndothermicityFactor*(bindingEnergy(index1)-H2_ON_H2_BINDING_ENERGY)/dustTemperature)
    END IF
    ! ENCOUNTER_DESORPTION_ENDOTHERMICITY_FACTOR = 0.75
    

    ! Rate of diffusion of index1 specie off of H2
    diffuseProb = vdiff(index1)*exp(-diffToBindRatio*H2_ON_H2_BINDING_ENERGY/dustTemperature)
    
    ! Rate of desorption of index1 specie off of H2
    desorbProb = vdiff(index1)*exp(-H2_ON_H2_BINDING_ENERGY/dustTemperature)

    ! Probability of desorbing when on top of H2
    desorbProb = desorbProb / (desorbProb + diffuseProb)

    ! Actual rate of EncounterDesorption mechanism
    EncounterDesorptionRate = 0.5D0*desorbProb* meetProb*GAS_DUST_DENSITY_RATIO/NUM_SITES_PER_GRAIN
END FUNCTION EncounterDesorptionRate

FUNCTION getNumberMonolayers(abundance) RESULT(numberMonolayers)
    REAL(dp) :: abundance, numberMonolayers
    
    IF (.NOT. GRAINS_HAVE_ICE) THEN
        numberMonolayers = 0.0d0
        RETURN
    END IF
    
    numberMonolayers = abundance * GAS_DUST_DENSITY_RATIO / NUM_SITES_PER_GRAIN
END FUNCTION

END MODULE SurfaceReactions
