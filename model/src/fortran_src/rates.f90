SUBROUTINE calculateReactionRates
    INTEGER:: idx1,idx2,k
    REAL(dp) :: vA,vB

    !Calculate all reaction rates
    !Assuming the user has temperature changes or uses the desorption features of phase 1,
    !these need to be recalculated every time step.

    idx1=crpReacs(1)
    idx2=crpReacs(2)
    IF (idx1 .ne. idx2) rate(idx1:idx2)=alpha(idx1:idx2)*zeta
    IF (improvedH2CRPDissociation) rate(nR_H2_CRP)=h2CRPRate

    !UV photons, radfield has (factor of 1.7 conversion from habing to Draine)
    idx1=photonReacs(1)
    idx2=photonReacs(2)
    IF (idx1 .ne. idx2) THEN
        rate(idx1:idx2) = alpha(idx1:idx2)*exp(-gama(idx1:idx2)*av(dstep))*radfield/1.7D0
    END IF

    !Reactions involving cosmic ray induced photon
    idx1=crphotReacs(1)
    idx2=crphotReacs(2)
    IF (idx1 .ne. idx2) THEN
        rate(idx1:idx2)=alpha(idx1:idx2)*gama(idx1:idx2)*1.0D0/(1.0D0-omega)*zeta*(gasTemp(dstep)/300D0)**beta(idx1:idx2)
    END IF

    !freeze out only happens if freezeFactor>0 and depending on evap choice 
    idx1=freezeReacs(1)
    idx2=freezeReacs(2)
    IF (idx1 .ne. idx2) THEN
        rate(idx1:idx2)=freezeOutRate(idx1,idx2)
        !freeze out rate uses thermal velocity but mass of E is 0 giving us infinite rates
        !just assume it's same as H
        rate(nR_EFreeze)=rate(nR_HFreeze)

        rate(nR_H2Freeze)=stickingCoefficient(h2StickingZero,h2StickingTemp,gasTemp(dstep))*rate(nR_H2Freeze)
        IF (h2StickingCoeffByh2Coverage) THEN
            ! If all surface is H2, (i.e. x_#H2 = safeMantle), assume no H2 sticks
            ! and so set the sticking coeff to 0. Linearly interpolate according to chance it will hit a H2 molecule on surface
            ! Perhaps there should also be a temperature term in here, with some boltzman of H2_ON_H2_BINDING_ENERGY and gasTemp.
            rate(nR_H2Freeze)=rate(nR_H2Freeze)*(1.0D0-abund(ngh2, dstep)/safeMantle)
        END IF
        
        rate(nR_HFreeze)=stickingCoefficient(hStickingZero,hStickingTemp,gasTemp(dstep))*rate(nR_HFreeze)
        if (hStickingCoeffByh2Coverage) THEN
            ! If all surface is H2, (i.e. x_#H2 = safeMantle), assume no H sticks
            ! and so set the sticking coeff to 0. Linearly interpolate according to chance it will hit a H2 molecule on surface
            ! Perhaps there should also be a temperature term in here, with some boltzman of H_ON_H2_BINDING_ENERGY and gasTemp.
            rate(nR_HFreeze)=rate(nR_HFreeze)*(1.0D0-abund(ngh2, dstep)/safeMantle)
        END IF
    END IF
    ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !The below desorption mechanisms are from Roberts et al. 2007 MNRAS with
    !the addition of direct UV photodesorption. DESOH2,DESCR1,DEUVCR
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    !Desorption due to energy released by H2 Formations
    idx1=desoh2Reacs(1)
    idx2=desoh2Reacs(2)
    IF (idx1 .ne. idx2) THEN
        IF ((desorb) .and. (h2desorb) .and. (safeMantle .gt. MIN_SURFACE_ABUND)) THEN
            !Epsilon is efficieny of this process, number of molecules removed per event
            !h2form is formation rate of h2, dependent on hydrogen abundance. 
            rate(idx1:idx2) = epsilon*h2FormEfficiency(gasTemp(dstep),dustTemp(dstep))

            !Don't remove species with binding energy > max BE removed by this process
            WHERE(gama(idx1:idx2) .gt. ebmaxh2) rate(idx1:idx2)=0.0D0
        ELSE
            rate(idx1:idx2) = 0.0D0
        ENDIF
        !turn off freeze out if desorption due to H2 formation is much faster
        !both rates combine with density to get rate of change so drop that factor
        WHERE((rate(freezePartners)*abund(re1(freezePartners),dstep))<&
        &MIN_SURFACE_ABUND*rate(idx1:idx2)) rate(freezePartners)=0.0D0
    END IF
    !Desorption due to energy from cosmic rays
    idx1=descrReacs(1)
    idx2=descrReacs(2)
    IF (idx1 .ne. idx2) THEN
        IF ((desorb) .and. (crdesorb) .and. (safeMantle .gt. MIN_SURFACE_ABUND)) THEN
            !4*pi*zeta = total CR flux. 1.64d-4 is iron to proton ratio of CR
            !as iron nuclei are main cause of CR heating.
            !GRAIN_SURFACEAREA_PER_H is the total surface area per hydrogen atom. ie total grain area per cubic cm when multiplied by density.
            !phi is efficieny of this reaction, number of molecules removed per event.
            rate(idx1:idx2) = 4.0D0*pi*zeta*1.64d-4*(GRAIN_SURFACEAREA_PER_H)*phi

            !Don't remove species with binding energy > max BE removed by this process
            WHERE(gama(idx1:idx2) .gt. ebmaxcr) rate(idx1:idx2)=0.0D0
        ELSE
            rate(idx1:idx2) = 0.0D0
        ENDIF
        !turn off freeze out if desorption due to CR formation is much faster
        WHERE((rate(freezePartners)*abund(re1(freezePartners),dstep)*density(dstep))&
        <MIN_SURFACE_ABUND*rate(idx1:idx2)) rate(freezePartners)=0.0D0
    END IF
    
    !Desorption due to UV, partially from ISRF and partially from CR creating photons
    idx1=deuvcrReacs(1)
    idx2=deuvcrReacs(2)
    IF (idx1 .ne. idx2) THEN
        IF ((desorb) .and. (uvdesorb) .and. (safeMantle .gt. MIN_SURFACE_ABUND)&
                &.and.(zeta .gt. 0)) THEN
            !4.875d3 = photon flux, Checchi-Pestellini & Aiello (1992) via Roberts et al. (2007)
            !UVY is yield per photon.
            rate(idx1:idx2) = GRAIN_CROSSSECTION_PER_H*uv_yield*4.875d3*zeta
            !additional factor accounting for UV desorption from ISRF. UVCREFF is ratio of 
            !CR induced UV to ISRF UV.
            rate(idx1:idx2) = rate(idx1:idx2) * (1.0D0+(radfield/uvcreff)*(1.0D0/zeta)*exp(-1.8D0*av(dstep)))

            !Don't remove species with binding energy > max BE removed by this process
            WHERE(gama(idx1:idx2) .gt. ebmaxuvcr) rate(idx1:idx2)=0.0D0
        ELSE
            rate(idx1:idx2) = 0.0D0
        ENDIF
        !turn off freeze out if desorption due to UV is much faster
        WHERE((rate(freezePartners)*abund(re1(freezePartners),dstep)*density(dstep))&
        &<MIN_SURFACE_ABUND*rate(idx1:idx2)) rate(freezePartners)=0.0D0
    END IF

    !CRS reactions represent the production of excited species from cosmic ray bombardment
    !rate equations from Shingledecker et. al. 2018
    idx1=crsReacs(1)
    idx2=crsReacs(2)
    IF (idx1 .ne. idx2) THEN
        !8.6 is the Spitzer-Tomasko cosmic ray flux in cm^-2 s^-1
        !1.3 converts to: ionisation rate/10^-17
        rate(idx1:idx2)=alpha(idx1:idx2)*(beta(idx1:idx2)*(gama(idx1:idx2)/100.0D0)*(8.6D0*zeta*1.3D0))
    END IF

    !EXRELAX, relaxation reactions for each excited species
    idx1=exrelaxReacs(1)
    idx2=exrelaxReacs(2)
    IF (idx1 .ne. idx2) THEN
        DO j=idx1,idx2
            DO i=lbound(iceList,1),ubound(iceList,1)
                IF (iceList(i) .eq. re1(j)) THEN
                    vA=vdiff(i)
                END IF
            END DO 
            rate(j)=vA
        END DO 
    END IF  

    !EXSOLID reactions represent the reactions of excited species on the grain
    idx1=exsolidReacs(1)
    idx2=exsolidReacs(2)

    IF (idx1 .ne. idx2) THEN
        !reaction rates calculated outside of UCLCHEM as per Shingledecker et al. 2018 and included in grain network
        !alpha are branching ratios and beta is reaction rate
        DO j=idx1,idx2
            DO i=lbound(iceList,1),ubound(iceList,1)
                IF (iceList(i) .eq. re1(j)) THEN
                    vA = vdiff(i)
                END IF
                IF (iceList(i) .eq. re2(j)) THEN
                    vB = vdiff(i)
                END IF
            END DO 
            rate(j) = (vB + vA)/(SURFACE_SITE_DENSITY*1.8d-8)
            rate(j) = alpha(j) * rate(j)
        END DO 
    END IF  

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Continuous Thermal Desorption. Reactions can be generated through a flag in Makerates
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    idx1=thermReacs(1)
    idx2=thermReacs(2)
    IF (idx1 .ne. idx2) THEN
        IF (thermdesorb) THEN
            DO j=idx1,idx2
                !then try to overwrite with position in grain array
                DO i=lbound(iceList,1),ubound(iceList,1)
                    !See Cuppen, Walsh et al. 2017 review (section 4.1)
                    IF (iceList(i) .eq. re1(j)) THEN
                        !Basic rate at which thermal desorption occurs
                        rate(j)=vdes(i)*exp(-gama(j)/dustTemp(dstep))
                        !factor of NUM_MONOLAYERS_IS_SURFACE adjusts for fact only top N monolayers (Eq 8)
                        !because GRAIN_SURFACEAREA_PER_H is per H nuclei, multiplying it by density gives area/cm-3
                        !that is roughly sigma_g.n_g from cuppen et al. 2017 but using surface instead of cross-sectional
                        !area seems more correct for this process.
                        IF (.NOT. THREE_PHASE) rate(j)=rate(j)*NUM_MONOLAYERS_IS_SURFACE*SURFACE_SITE_DENSITY*GRAIN_SURFACEAREA_PER_H
                    END IF
                END DO
            END DO
            
            ! ! Testing, does not fix stiffness problem at higher densities
            ! WHERE(ABS(rate(freezePartners)*abund(re1(freezePartners),dstep)*density(dstep) - &
            !         &re1(idx1:idx2) * rate(idx1:idx2))/re1(idx1:idx2) < 1.0D-20)
            !     rate(idx1:idx2) = 0.0D0
            !     rate(freezePartners) = 0.0D0
            ! END WHERE

            !At some point, rate is so fast that there's no point freezing out any more
            !Save the integrator some trouble and turn freeze out off
            WHERE(rate(freezePartners)*abund(re1(freezePartners),dstep)*density(dstep)&
                &<MIN_SURFACE_ABUND*rate(idx1:idx2)) rate(freezePartners)=0.0D0
        

            IF (safeMantle .lt. MIN_SURFACE_ABUND) rate(idx1:idx2)=0.0D0
        ELSE
            rate(idx1:idx2)=0.0D0
        END IF
    END IF

    ! Number of monolayers of ice in total (surface + bulk)
    numMonolayers = getNumberMonolayers(safeMantle + safeBulk)

    !Reactions on surface can be treated considering diffusion of reactants
    !as in Langmuir-Hinshelwood mechanism
    !See work of David Quenard 2017 Arxiv:1711.05184
    !First calculate rate of the diffusion reaction
    idx1=lhReacs(1)
    idx2=lhReacs(2)
    if (idx1 .ne. idx2) THEN
        if ((dustTemp(dstep) .lt. MAX_GRAIN_TEMP) .and. (safeMantle .gt. MIN_SURFACE_ABUND)) THEN
            DO j=idx1,idx2
                rate(j)=diffusionReactionRate(j,dustTemp(dstep))
            END DO
            
            IF ((desorb) .and. (chemdesorb)) THEN
                !two routes for every diffusion reaction: products to gas or products remain on surface

                !calculate fraction of reaction that goes down desorption route
                idx1=lhdesReacs(1)
                idx2=lhdesReacs(2)
                k = 0
                DO i=idx1, idx2
                    k = k + 1
                    rate(i)=desorptionFractionIncludingIce(i, numMonolayers)*rate(LHDEScorrespondingLHreacs(k))
                    ! IF ((trim(specname(re1(i))) .eq. "#SO2") .and. (trim(specname(re2(i))) .eq. "#C")) THEN
                    !     WRITE(*,*) "REACTION:"
                    !     WRITE(*,*) specname(re1(i)), specname(re2(i)), &
                    !         specname(p1(i)), specname(p2(i))
                    !     WRITE(*,*) "DESORPTION FRACTION INCL ICE", desorptionFractionIncludingIce(i, numMonolayers)
                    !     WRITE(*,*) "CORRESPONDING LH REACTION RATE", rate(LHDEScorrespondingLHreacs(k))
                    !     WRITE(*,*) "RESULTING LHDES REACTION RATE", rate(i)
                    ! END IF
                END DO
                
                !remove that fraction from total rate of the diffusion route
                k = 0
                DO i = idx1, idx2
                    k = k + 1
                    rate(LHDEScorrespondingLHreacs(k)) = rate(LHDEScorrespondingLHreacs(k)) - rate(i)
                    ! IF ((trim(specname(re1(i))) .eq. "#SO2") .and. (trim(specname(re2(i))) .eq. "#C")) THEN
                    !     WRITE(*,*) "SUBTRACTING REACTION:"
                    !     WRITE(*,*) specname(re1(i)), specname(re2(i)), &
                    !         specname(p1(i)), specname(p2(i))
                    !     WRITE(*,*) "LHDES REACTION RATE", rate(i)
                    !     WRITE(*,*) "CORRESPONDING LH REACTION RATE", rate(LHDEScorrespondingLHreacs(k))
                    ! END IF

                END DO
                ! WRITE(*,*) "FINISHED CORRECTING REACTION RATE OF LH REACTION"
                ! (we cannot do this in the first loop through lhdesReacs, because then if we have a two-product reaction,
                !  the second reaction we find would have its rate decreased already by the first one)
            ELSE
                rate(lhdesReacs(1):lhdesReacs(2))=0.0D0
            END IF
        ELSE
            rate(idx1:idx2)=0.0D0
            rate(lhdesReacs(1):lhdesReacs(2))=0.0D0
        END IF
    END IF

    !Account for Eley-Rideal reactions in a similar way.
    !First calculate overall rate and then split between desorption and sticking
    idx1=erReacs(1)
    idx2=erReacs(2)
    if (idx1 .ne. idx2) THEN
        rate(idx1:idx2)=freezeOutRate(idx1,idx2)
        rate(idx1:idx2)=rate(idx1:idx2)*exp(-gama(idx1:idx2)/dustTemp(dstep))
        
        IF ((desorb) .and. (chemdesorb) .and. (safeMantle .gt. MIN_SURFACE_ABUND)) THEN
            !two routes for every diffusion reaction: products to gas or products remain on surface

            !calculate fraction of reaction that goes down desorption route
            idx1 = erdesReacs(1)
            idx2 = erdesReacs(2)
            k = 0
            DO i=idx1, idx2
                k = k + 1
                rate(i)=desorptionFractionIncludingIce(i, numMonolayers)*rate(ERDEScorrespondingERreacs(k))
            END DO
            
            !remove that fraction from total rate of the diffusion route
            k = 0
            DO i = idx1, idx2
                k = k + 1
                rate(ERDEScorrespondingERreacs(k)) = rate(ERDEScorrespondingERreacs(k)) - rate(i)
            END DO
            ! (we cannot do this in the first loop through erdesReacs, because then if we have a two-product reaction,
            !  the second reaction we find would have its rate decreased already by the first one)
        END IF
    END IF

    IF (PARAMETERIZE_H2FORM) THEN
        rate(nR_H2Form_CT)=h2FormEfficiency(gasTemp(dstep),dustTemp(dstep))
        !rate(nR_H2Form_LH)=0.0D0
        rate(nR_H2Form_ER)=0.0D0
        !rate(nR_H2Form_LHDes)=0.0D0
        rate(nR_H2Form_ERDes)=0.0D0
    ELSE
        rate(nR_H2Form_CT)= 0.0D0
    END IF

    CALL bulkSurfaceExchangeReactions(rate,dustTemp(dstep))
    
    !Basic gas phase reactions 
    !They only change if temperature has so we can save time with an if statement
    idx1=twobodyReacs(1)
    idx2=twobodyReacs(2)
    IF (lastGasTemp .ne. gasTemp(dstep)) THEN
        rate(idx1:idx2) = alpha(idx1:idx2)*((gasTemp(dstep)/300.0D0)**beta(idx1:idx2))*exp(-gama(idx1:idx2)/gasTemp(dstep)) 
    END IF

    idx1=ionopol1Reacs(1)
    idx2=ionopol1Reacs(2)
    IF ((idx1 .ne. idx2) .and. (lastGasTemp .ne. gasTemp(dstep))) THEN
        !This formula including the magic numbers come from KIDA help page.
        rate(idx1:idx2)=alpha(idx1:idx2)*beta(idx1:idx2)*(0.62d0+0.4767d0*gama(idx1:idx2)*sqrt(300.0d0/gasTemp(dstep)))
    END IF

    idx1=ionopol2Reacs(1)
    idx2=ionopol2Reacs(2)
    IF ((idx1 .ne. idx2) .and. (lastGasTemp .ne. gasTemp(dstep))) THEN
        !This formula including the magic numbers come from KIDA help page.
        rate(idx1:idx2)=alpha(idx1:idx2)*beta(idx1:idx2)*(1.0d0+0.0967d0*gama(idx1:idx2)&
        &*sqrt(300.0d0/gasTemp(dstep))+gama(idx1:idx2)*gama(idx1:idx2)*300.0D0/(10.526*gasTemp(dstep)))
    END IF
    lastGasTemp=gasTemp(dstep)
    lastDustTemp=dustTemp(dstep)

    !turn off reactions outside their temperature range
    WHERE(gasTemp(dstep) .lt. minTemps) rate=0.0D0

    WHERE(gasTemp(dstep) .gt. maxTemps) rate=0.0D0

    !Overwrite reactions for which we have a more detailed photoreaction treatment
    rate(nR_H2_hv)=H2PhotoDissRate(h2Col,radField,av(dstep),turbVel)!H2 photodissociation
    rate(nR_CO_hv)=COPhotoDissRate(h2Col,coCol,radField,av(dstep)) !CO photodissociation
    rate(nR_C_hv)=cIonizationRate(alpha(nR_C_hv),gama(nR_C_hv),gasTemp(dstep),ccol,h2col,av(dstep),radfield) !C photoionization
    
    IF ((h2EncounterDesorption) .and. (safeMantle .gt. MIN_SURFACE_ABUND)) THEN
        rate(nR_H2_ED)=EncounterDesorptionRate(nR_H2_ED, dustTemp(dstep)) !H2 Encounter Desorption Hincelin 2015
    ELSE
        rate(nR_H2_ED)=0.0D0
    END IF

    IF ((hEncounterDesorption) .and. (safeMantle .gt. MIN_SURFACE_ABUND)) THEN
        ! Not quite correct, still uses H2_ON_H2_BINDING_ENERGY
        rate(nR_H_ED)=EncounterDesorptionRate(nR_H_ED, dustTemp(dstep)) !H2 Encounter Desorption Hincelin 2015
    ELSE
        rate(nR_H_ED)=0.0D0
    END IF

END SUBROUTINE calculateReactionRates



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!Freeze out determined by rate of collisions with grain
!No sticking coefficient is used because typical values are >0.95 below 150 K
! eg Le Bourlot et al. 2013, Molpeceres et al. 2020
!Above 150 K, thermal desorption will completely remove grain species
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION freezeOutRate(idx1,idx2) RESULT(freezeRates)
    REAL(dp) :: freezeRates(idx2-idx1+1)
    INTEGER :: idx1,idx2
    
    !additional factor for ions (beta=0 for neutrals)
    freezeRates=1.0D0+beta(idx1:idx2)*16.71d-4/(GRAIN_RADIUS*gasTemp(dstep))
    IF ((freezeFactor .eq. 0.0) .or. (dustTemp(dstep) .gt. MAX_GRAIN_TEMP)) then
        freezeRates=0.0D0
    ELSE
        freezeRates=freezeRates*freezeFactor*alpha(idx1:idx2)*THERMAL_VEL&
        &*sqrt(gasTemp(dstep)/mass(re1(idx1:idx2)))*GRAIN_CROSSSECTION_PER_H
    END IF

END FUNCTION freezeOutRate


FUNCTION stickingCoefficient(stickingZero,criticalTemp,gasTemp) RESULT(stickingCoeff)
    !Sticking coefficient for freeze out taken from Chaabouni et al. 2012 A&A 538 Equation 1
    REAL(dp) :: stickingCoeff
    REAL(dp) :: stickingZero,criticalTemp,gasTemp,tempRatio
    REAL(dp) :: beta=2.5d0
    tempRatio=gasTemp/criticalTemp
    
    stickingCoeff=stickingZero*(1.0d0+beta*tempRatio)/((1.0d0+tempRatio)**beta)
END FUNCTION stickingCoefficient
