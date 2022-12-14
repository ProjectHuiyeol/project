const qnaList = [
  {
    q: '1. 하루 중 컨디션이 제일 좋은 시간은?',
    a: [
      { answer: 'a. 아침', score: [0, 10, 0] },
      { answer: 'b. 낮부터 이른 저녁', score: [10, 0, 0] },
      { answer: 'c. 늦은 밤', score: [0, 0, 10] }
    ]
  },
  {
    q: '2. 보통 어떻게 걸으시나요?',
    a: [
      { answer: 'a. 큰 보폭으로 빨리', score: [10, 0, 0] },
      { answer: 'b. 고개를 들고 보통 속도로', score: [0, 10, 0] },
      { answer: 'c. 느린 속도로', score: [0, 0, 10] }
    ]
  },
  {
    q: '3. 사람들과 대화할 때 당신의 제스처는?',
    a: [
      { answer: 'a. 팔짱을 끼고 있다', score: [0, 0, 10] },
      { answer: 'b. 대화하는 사람을 툭툭 치거나 건드린다', score: [10, 0, 0] },
      { answer: 'c. 귀나 턱, 머리카락을 만진다', score: [0, 10, 0] }
    ]
  },
  {
    q: '4. 앉아서 쉴 때 어떻게 앉으시나요?',
    a: [
      { answer: 'a. 다리를 가지런히 하고 단정히 앉는다', score: [0, 0, 10] },
      { answer: 'b. 다리를 꼬고 앉는다', score: [0, 10, 0] },
      { answer: 'c. 다리를 쭉 뻗고 앉는다', score: [10, 0, 0] }
    ]
  },
  {
    q: '5. 정말 즐거운 일이 생겼을 때 어떻게 반응하시나요?',
    a: [
      { answer: 'a. 아주 크게 웃는다', score: [10, 0, 0] },
      { answer: 'b. 웃긴 하되 엄청 크게 웃진 않는다', score: [0, 10, 0] },
      { answer: 'c. 조용히 미소 짓는다', score: [0, 0, 10] }
    ]
  },
  {
    q: '6. 파티나 모임에 갔을 때 어떻게 행동하시나요?',
    a: [
      {
        answer: 'a. 임팩트 있게 등장해 모두 내가 온 걸 알아차리게 한다',
        score: [10, 0, 0]
      },
      {
        answer: 'b. 조용히 들어가서 아는 사람이 있는지 둘러본다',
        score: [0, 10, 0]
      },
      {
        answer: 'c. 최대한 조용히 들어가서 아무도 내가 있는지 모르게 한다',
        score: [0, 0, 10]
      }
    ]
  },
  {
    q: '7. 초집중하고 있는 당신에게 누군가 끼어들거나 말을 건다면?',
    a: [
      { answer: 'a. 휴식이 반갑다', score: [0, 10, 0] },
      { answer: 'b. 너무 짜증난다', score: [10, 0, 0] },
      { answer: 'c. 상황에 따라 다르다', score: [0, 0, 10] }
    ]
  },
  {
    q: '8. 여행을 떠났을 때 가장 듣고 싶은 노래는??',
    a: [
      { answer: 'a. 붉은 노을이 생각나는 노래', score: [10, 0, 0] },
      { answer: 'b. 어두운 밤하늘이 떠오르는 노래', score: [0, 0, 10] },
      { answer: 'c. 발랄하고 청량한 노래', score: [10, 10, 0] },
      { answer: 'd. 싱그러운 숲속에 있는듯한 노래', score: [0, 10, 0] },
      { answer: 'e. 몽환적인 노래', score: [10, 0, 10] },
      { answer: 'f. 맑고 깨끗한 느낌의 노래', score: [0, 10, 10] }
    ]
  },
  {
    q: '9. 밤에 잠들기 직전에 어떻게 눕나요?',
    a: [
      { answer: 'a. 천장을 보고 똑바로 눕는다', score: [0, 10, 0] },
      { answer: 'b. 배를 바닥에 깔고 엎드린다', score: [10, 0, 0] },
      { answer: 'c. 머리 끝까지 이불을 덮는다', score: [0, 0, 10] }
    ]
  },
  {
    q: '10. 어떤 꿈을 자주 꾸시나요?',
    a: [
      { answer: 'a. 싸우는 꿈', score: [10, 0, 0] },
      { answer: 'b. 사람이나 물건을 찾는 꿈', score: [0, 10, 0] },
      { answer: 'c. 하늘을 날거나 물 위에 떠다니는 꿈', score: [0, 0, 10] },
      { answer: 'd. 거의 꿈을 안 꾼다', score: [0, 0, 0] },
      { answer: 'e. 항상 좋은 꿈만 꾼다', score: [10, 10, 10] }
    ]
  }
]

const infoList = [
  {
    from: 10,
    to: 20,
    mLeft: '6%',
    name: '민트',
    desc: '당신의 이미지 컬러는 로맨틱한 민트. 감수성이 풍부한 당신은 예술에도 남다른 재능이 있네요. 직감이 뛰어나 사람들의 기분에 잘 맞춰주는 재주도 있습니다. 쾌락을 추구하는 욕구가 강한 편이어서 연애 같은 것에 녹아 내리듯 빠져들기 쉬운 타입이지만, 그만큼 직감이 뛰어나 이 점을 잘 살린다면 일에서도 남들이 보지 못하는 해법과 기회를 늘 발견할 수 있겠네요. 이 뛰어난 직관력으로 사람들의 기분도 쉽게 파악하는 편일거에요. 당신은 아무것도 하지 않고 멍하니 있을 때야말로, 뜻밖의 아이디어가 차례차례로 떠오르는 타입이죠. 무의식적인 직감이 강한 타입이라, 이래저래 머릿속으로 재기 보다 솔직한 직감에 맡기는 편이 의외로 올바른 선택을 유도하기도 할거에요. 그런 감성을 살려 그림을 그리거나 글을 쓴다면 훌륭한 작품이 탄생할 수 있을지도 모르겠습니다.'
  },
  {
    from: 21,
    to: 30,
    mLeft: '12%',
    name: '블랙',
    desc: '당신의 이미지 컬러는 밤하늘 같은 검정. 쿨하고 차분한 당신은 매사에 냉철하고 공정한 판단을 내리는 편입니다. 그런 뛰어난 판단력과 어른스러움이 주변 사람들을 끄는 매력이네요. 주변 사람들이 어려운 일이 있을 때 당신의 뛰어난 판단력을 믿고 자주 조언을 구하지는 않는지요? 당신은 무엇이 플러스가 되고 무엇이 마이너스가 될지, 쉽게 골라잡을 수 있는 타입입니다. 또한 문제의 핵심이 어디에 있는지, 그 진실을 꿰뚫어 볼 수 있어 필요한 것은 유지하고, 불필요한 것은 단호하게 끊어버리는 판단력이 매우 강한 사람입니다. 당신은 이 신중함으로 한 번 몰두한 것에 대해서 성실히 응하게 되어, 사람들로부터 ‘성의가 있는 사람’이라 평가를 받을 것입니다. 지금 당신의 노력은 반드시 높은 결과로 보답할 것입니다. 그러므로 언제나 스스로 기합을 넣고 열심히 해봅시다.'
  },
  {
    from: 31,
    to: 40,
    mLeft: '18%',
    name: '화이트',
    desc: '당신의 이미지 컬러는 눈 내린 하얀색. 막 내린 눈처럼 포근하고 다정한 당신은 다른 사람을 늘 배려하는 속 깊은 사람이네요. 사랑하는 존재는 반드시 지키려고 하는 의외의 강인함도 갖고 있습니다. 당신은 주변 사람들을 밝고 따뜻하게 대하기에 모두와 사이 좋고 즐겁게 지낼 수 있습니다. 일의 경우에도 혼자 하는 것 보다는 누군가와 협력해서 하는 편이 더 좋은 결과를 낳을 것이며 자신도 선호하는 방향입니다. 충실한 마음을 모두와 나눠 가져주세요. 고민이 있다면 가까운 누군가에게 상담해보세요. 다정한 조언을 얻을 수 있을 것입니다.'
  },
  {
    from: 41,
    to: 50,
    mLeft: '24%',
    name: '핑크',
    desc: '당신의 이미지 컬러는 4월의 벚꽃같은 핑크입니다. 늘 애정으로 가득찬 당신은 감수성이 강해서 쉽게 감동받는 소녀처럼 아름다운 사람이네요. 그런 순수함이 당신의 매력포인트입니다. 당신의 풍부한 감수성 때문에 어릴적 글과 예술에 소질이 있지는 않았나요? 당신은 쾌락에 이끌려 기분이 풀어져서 다소 해야 할 일에 대해 게을러지기 쉬운 면도 있지만, 이러한 풍부한 감수성과 직감이 당신의 엄청난 강점이니 절대 외면하지 마세요. 당신은 남들보다 감정이 풍부하기 때문에, 책을 읽거나 영화를 보거나 음악을 들으면 가슴 한 구석이 울리는 것을 느낄 때가 자주 있을거에요.'
  },
  {
    from: 51,
    to: 60,
    mLeft: '30%',
    name: '실버',
    desc: '당신의 이미지 컬러는 활력이 넘치는 실버. 단체에선 늘 주도권을 지는 리더적 성향을 지닌 사람입니다. 마음 속의 뜨거운 열정과 의지를 바탕으로 뭐든 이뤄내는 능력을 지녔네요. 당신은 알고보면 의지가 확고한 사람입니다. 지금 당신이 마음속에 품고 있는 이상이나 목표를 믿고 나아간다면, 반드시 성공적인 앞길이 열릴 것입니다. 당신은 자신감으로 넘쳐흐르고 있으므로 자신을 믿는다면 무엇을 하든 힘을 충분히 발휘할 수 있습니다.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '36%',
    name: '보라',
    desc: '당신의 이미지 컬러는 황제의 색 보라. 당신은 늘 새로운 것을 추구하며, 일상 속에서 자극을 찾는 사람입니다. 항상 더 넓은 세계로 향하는 것을 꿈꾸는 편이군요. 당신은 남들보다 넓은 시야로 여러 가지를 알고 싶다는 욕망이 끓어오르는 사람입니다. 항상 매너리즘을 싶어하며, 새로운 세계로 뛰어 들 사람입니다. 당신 스스로는 호기심이 강해 가끔은 충동적으로 행동하고 싶어하는 사람입니다. 그 마음으로 평소 당신의 패턴과 생활권 밖으로 향한다면, 의외의 동료와 친구를 사귀고 항상 새로운 기회를 맞이하게 될거에요.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '42%',
    name: '연보라',
    desc: '당신의 이미지 컬러는 빗 속의 수국같은 연보라. 마치 수국처럼 환경에 따라 자신의 색을 바꿀 수 있는 강한 적응력이 당신의 무기입니다. 또한 특유의 예리함으로 풍부한 발상을 하는 점도 장점이네요. 당신은 항상 사소한 ‘번뜩임’이 찾아와 풍부한 발상이나 아이디어를 내는 사람입니다. 당신이 무슨 일을 하든 기획하거나 계획을 세우려 한다면, 번뜩이는 아이디어가 속속 찾아 올 것입니다. 당신은 타인과는 다른 관점으로 매사를 바라볼 수 있어, 조금만 고민한다면 막혀있던 것을 새로운 방식으로 풀어나가는 사람이네요.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '48%',
    name: '레드',
    desc: '당신의 이미지 컬러는 열정적인 붉은 장미의 빨강입니다. 늘 열정과 의욕으로 가득하고 새로운 것에 도전하는 것을 즐깁니다. 일이 막히더라도 긍정적인 마인드로 풀어가는 편이네요. 한가지 일에 빠지면 큰 열정과 집중력을 보이지 않는지요? 사람과의 관계도 좋아하는 사람들에게 집중하는 형태일거라 생각되네요. 숨겨진 당신은 사실에 대해 더 알고 싶고, 좀 더 깊은 이면까지 파고들고 싶다는 마음이 강해, 자연히 지금 껴안고 있는 일이나 문제에 깊이 관여하게 됩니다. 그 문제와 꼭 엮여야만 한다는 기분이 강하게 샘솟고 있는 듯 하군요. 당신은 항상 사람들과 깊게 엮이며 뭔가 변화를 일으킬 것 같습니다. 주변 사람들이 하는 말을 잘 듣고 있다 보면 당신의 마음을 움직이는 어떤 것이 있을 것입니다.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '54%',
    name: '하늘색',
    desc: '당신의 이미지 컬러는 비 개인 하늘색입니다. 비 온 뒤 하늘처럼 머리가 맑고, 생각이 깊은 사람이네요. 꼼꼼한 부분까지 늘 신경쓰고 자기관리도 철저한 편입니다. 남들보다 사소한 것에 눈을 두기 쉽고, 디테일을 잘 보기 때문에 그 성향을 잘 살린다면 타인의 세세한 부분까지 배려할 수 있으며, 일에서도 큰 성공을 거둘 수 있는 타입입니다. 인간관계에서는 매우 가까운 사람들이 아니면 감정을 상대방에게 잘 전달하지 못할 수도 있지만, 스스로를 지나치게 몰아 세우지 말고 릴렉스한다면 남들의 조언과 공감도 쉽게 얻어내는 타입이네요.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '60%',
    name: '노란색',
    desc: '당신의 이미지 컬러는 귀여운 노랑입니다. 마치 아기 오리처럼 호기심이 왕성하고 활기로 가득차 있네요. 그런 적극성이 당신의 미래를 밝게 열어 줄 것입니다. 당신은 다양한 사람과 이야기하거나, 이곳 저곳을 쏘다니며 당신의 관심사에 대해 머리 쓰기를 좋아하는 사람이네요. 동시에 타인과의 커뮤니케이션 능력도 강하기에, 끊임없이 일과 일상의 변화를 모색하는 편이네요. 또한 기력으로 가득 차 매사를 남들보다 넓은 시야로 보는 사람입니다.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '66%',
    name: '초록색',
    desc: '당신의 이미지 컬러는 한여름 숲 속같은 초록입니다. 차분하고 성실한 당신은 안정을 추구하는 편이며 꾸준히 노력하는 것을 미덕으로 삼는 사람입니다. 당신의 차분하고 끈기 있는 성향이 앞으로 분명 인생을 성공적으로 이끌어 줄거에요. 단지 가끔 자기중심적으로 생각하기 쉬운 경향도 있지만, 이런 당신의 성향이 당신을 한 가지 일에 집중해서 꾸준히 노력하여 성공을 이뤄내도록 해주는 요소입니다. 지금 확실히 노력해둔다면, 일이나 학업도 곧 열매를 맺고, 반드시 성공적인 결과를 도출해낼 것입니다. 여러 가지 일에 조금씩 손대는 것 보다, 좋아하는 한 가지 일에 꾸준히 몰두해주세요. 당신이 노력해서 얻은 것은 당신의 인생에서 행운을 가져다 줄 것입니다.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '72%',
    name: '베이비 핑크',
    desc: '당신의 이미지 컬러는 순진무구한 베이비 핑크. 당신은 남들에게 어리광 부리기를 좋아하는 귀여운 사람이네요. 붙임성이 좋고 순진무구한 마음을 지닌 당신은 어딜가든 환영받는 존재입니다. 당신은 주변 사람들과의 감정적인 유대를 소중히 하는 편이네요. 당신은 보통 사람보다 가까이 있는 사람의 기분을 잘 느끼며, 모두와 공감하면서 마음이 통하는 교류에 뛰어납니다. 또한 감수성이 풍부해서 그림을 그리거나 글쓰기에 도전해본다면 좋은 작품이 나올지도 모르겠네요.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '78%',
    name: '남색',
    desc: '당신의 이미지 컬러는 청춘의 남색. 당신은 감정 표현이 풍부합니다. 당신은 늘 마음 속 어딘가에서 타인과 감정적으로 따뜻한 교류를 원하는 사람입니다. 당신은 남들보다 감정이 매우 유들유들하므로 있으므로 가까이 있는 사람들과 즐거운 시간을 보낼 수 있습니다. 당신의 풍부한 감정과 감성으로 예술이나 글쓰기를 해보면 어떨까요? 의외로 엄청난 선천적 소질을 보일지 모르는 타입입니다.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '84%',
    name: '골드',
    desc: '당신의 이미지 컬러는 희망의 골드. 확고한 의지를 통해 자신의 미래를 스스로 개척하는 능력을 지닌 사람입니다. 늘 자발적인 당신은 주변 사람에게 모범이 되는 존재입니다. 이제부터 나아갈 방향을 확실히 정하고 그것을 향해 나아가려 시도하면, 지금 마주하고 있는 문제의 현상을 좀 더 좋은 방향으로 바꿀 수 있을 것입니다. 지금 당신이 마음속에 품고 있는 이상이나 목표를 믿고 나아간다면, 반드시 성공적인 앞길이 열릴 것입니다. 당신은 자신감으로 넘쳐흐르고 있으므로 무엇을 하든 자신의 힘을 충분히 발휘할 수 있겠습니다.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '90%',
    name: '코랄',
    desc: '당신의 이미지 컬러는 통통튀는 코랄입니다. 늘 자신감과 에너지로 가득찬 당신은 활동적으로 움직이는 것을 좋아합니다. 새로운 것에 도전하는 것도 즐기는 편이네요. 당신에게는 건설적인 에너지가 용솟음 치고 있습니다. 뭔가 새로운 것을 만들고 싶어하는 욕구도 무척 강한 편이네요. 주변 사람들에게 아이디어도 풍부하고, 새로운 힌트를 끊임없이 제공한다는 평가를 듣지는 않으시나요? 이 뜨거운 기분이 결코 흔들리는 법도 없고, 스스로에 대한 자신감도 가득하네요. 당신의 이런 마음을 좀 더 달아오르게 해서, 정열적으로 활동성을 드러내고자 하는 편이 좋습니다. 당신은 직감이 살아있으며, 그 직감을 행동으로 옮길 때 혼자서 하기 보다 사람들에게 도움을 받아 협업한다면 더 효과적으로 일을 진행시킬 수 있을거에요.'
  },
  {
    from: 61,
    to: 70,
    mLeft: '96%',
    name: '갈색',
    desc: '당신의 이미지 컬러는 포근한 원목 가구같은 갈색입니다. 매사에 신중하고 착실한 당신의 강점은 강한 책임감 입니다. 그런 당신을 정신적 지주로 생각하는 사람들이 많을거에요. 이런 진실된 수수함이야 말로 당신이 일상 속에 잘 파고들어 일을 오랫동안 계속해 나갈 수 있다는 것을 의미합니다. 지금 마주하고 있는 문제나 일이 다소 막히고 큰 진전은 없을지언정, 당신의 진실됨과 착실함으로 결국 큰 성공을 이뤄낼거라 자신합니다. 당신은 남들보다 신중한 타입이라 일 하나하나에 책임감을 갖고 임하는 성향입니다. 언제나 실력과 기초를 단단히 쌓는데 힘을 쏟는다면 일과 커리어에서도 큰 성공을 거둘 수 있는 대기만성형 타입입니다.'
  }
]
